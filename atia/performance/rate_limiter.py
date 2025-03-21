"""
Rate Limiter for controlling request rates.

This module provides rate limiting capabilities using token bucket algorithm
to prevent API abuse and ensure fair resource allocation.
"""

import asyncio
import logging
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable, Tuple
import functools

from atia.config import settings
from atia.performance.models import RateLimitRule, RateLimitState
from atia.utils.error_handling import catch_and_log


logger = logging.getLogger(__name__)


class RateLimitExceededError(Exception):
    """Error raised when rate limit is exceeded."""
    pass


class TokenBucketRateLimiter:
    """
    Implements the token bucket algorithm for rate limiting.
    """

    def __init__(self, 
                requests_per_second: float, 
                burst_size: int = 10,
                rule_id: Optional[str] = None):
        """
        Initialize the token bucket rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size (tokens in the bucket)
            rule_id: Optional rule ID
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.rule_id = rule_id or f"rule_{int(time.time())}"
        self.tokens = burst_size  # Start with a full bucket
        self.last_updated = time.time()
        self.total_requests = 0
        self.total_limited = 0

    async def acquire(self, tokens: float = 1.0, wait: bool = False) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            wait: Whether to wait for tokens to become available

        Returns:
            True if tokens were acquired, False otherwise
        """
        # Refresh tokens based on time elapsed
        self._refresh()

        # Check if enough tokens are available
        if self.tokens >= tokens:
            self.tokens -= tokens
            self.total_requests += 1
            return True
        elif wait:
            # Calculate time to wait for enough tokens
            time_to_wait = (tokens - self.tokens) / self.requests_per_second

            # Sleep for that duration
            await asyncio.sleep(time_to_wait)

            # Tokens should be available now
            self._refresh()
            self.tokens -= tokens
            self.total_requests += 1
            return True
        else:
            # Not enough tokens and not waiting
            self.total_limited += 1
            return False

    def _refresh(self):
        """Refresh tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_updated

        # Calculate new tokens to add
        new_tokens = elapsed * self.requests_per_second

        # Add tokens, up to burst size
        self.tokens = min(self.burst_size, self.tokens + new_tokens)
        self.last_updated = now

    def get_state(self) -> RateLimitState:
        """Get the current state of the rate limiter."""
        self._refresh()  # Update tokens before returning state

        return RateLimitState(
            rule_id=self.rule_id,
            tokens=self.tokens,
            last_updated=datetime.fromtimestamp(self.last_updated),
            total_requests=self.total_requests,
            total_limited=self.total_limited
        )


class AdaptiveRateLimiter(TokenBucketRateLimiter):
    """
    Rate limiter that can adapt based on system load or error rates.
    """

    def __init__(self, 
                base_requests_per_second: float, 
                min_requests_per_second: float,
                max_requests_per_second: float,
                burst_size: int = 10,
                rule_id: Optional[str] = None):
        """
        Initialize the adaptive rate limiter.

        Args:
            base_requests_per_second: Base maximum requests per second
            min_requests_per_second: Minimum requests per second when throttling
            max_requests_per_second: Maximum possible requests per second
            burst_size: Maximum burst size (tokens in the bucket)
            rule_id: Optional rule ID
        """
        super().__init__(base_requests_per_second, burst_size, rule_id)
        self.base_requests_per_second = base_requests_per_second
        self.min_requests_per_second = min_requests_per_second
        self.max_requests_per_second = max_requests_per_second

        # Stats for adaptation
        self.error_count = 0
        self.success_count = 0
        self.adaptation_window = 10  # Number of requests to consider for adaptation
        self.adaptation_threshold = 0.2  # Error ratio threshold for throttling

    async def acquire(self, tokens: float = 1.0, wait: bool = False, 
                     success: Optional[bool] = None) -> bool:
        """
        Acquire tokens and update stats for adaptation.

        Args:
            tokens: Number of tokens to acquire
            wait: Whether to wait for tokens to become available
            success: Whether the previous operation was successful

        Returns:
            True if tokens were acquired, False otherwise
        """
        # Update stats if success is provided
        if success is not None:
            if success:
                self.success_count += 1
            else:
                self.error_count += 1

            # Check if we need to adapt
            total = self.success_count + self.error_count
            if total >= self.adaptation_window:
                self._adapt()

        # Use parent implementation
        return await super().acquire(tokens, wait)

    def _adapt(self):
        """Adapt the rate limit based on error ratio."""
        total = self.success_count + self.error_count
        if total == 0:
            return

        error_ratio = self.error_count / total

        if error_ratio > self.adaptation_threshold:
            # Too many errors, throttle down
            self.requests_per_second = max(
                self.min_requests_per_second,
                self.requests_per_second * 0.8  # Reduce by 20%
            )
            logger.info(f"Adapting rate limit down to {self.requests_per_second} req/s due to high error rate")
        else:
            # Few errors, can increase gradually
            self.requests_per_second = min(
                self.max_requests_per_second,
                self.requests_per_second * 1.05  # Increase by 5%
            )
            logger.debug(f"Adapting rate limit up to {self.requests_per_second} req/s due to low error rate")

        # Reset stats
        self.error_count = 0
        self.success_count = 0


class RateLimiterRegistry:
    """
    Registry for managing multiple rate limiters.
    """

    def __init__(self, storage_dir: str = "data/rate_limiters"):
        """
        Initialize the rate limiter registry.

        Args:
            storage_dir: Directory to store rate limiter state
        """
        self.storage_dir = storage_dir
        self.rate_limiters: Dict[str, TokenBucketRateLimiter] = {}
        self.rules: Dict[str, RateLimitRule] = {}

        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)

        # Load rules from storage
        self._load_rules()

    @catch_and_log(component="rate_limiter_registry")
    def add_rule(self, rule: RateLimitRule) -> str:
        """
        Add a rate limit rule.

        Args:
            rule: The rate limit rule to add

        Returns:
            Rule ID
        """
        # Store the rule
        self.rules[rule.id] = rule

        # Create a rate limiter for this rule
        if rule.is_enabled:
            self.rate_limiters[rule.id] = TokenBucketRateLimiter(
                requests_per_second=rule.requests_per_second,
                burst_size=rule.burst_size,
                rule_id=rule.id
            )

        # Save the rules
        self._save_rules()

        return rule.id

    @catch_and_log(component="rate_limiter_registry")
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a rate limit rule.

        Args:
            rule_id: ID of the rule to update
            updates: Dictionary of updates to apply

        Returns:
            True if successful, False if rule not found
        """
        if rule_id not in self.rules:
            return False

        # Get the current rule
        rule = self.rules[rule_id]

        # Update fields
        for field, value in updates.items():
            if hasattr(rule, field):
                setattr(rule, field, value)

        # Update rate limiter if needed
        if "requests_per_second" in updates or "burst_size" in updates or "is_enabled" in updates:
            if rule.is_enabled:
                self.rate_limiters[rule_id] = TokenBucketRateLimiter(
                    requests_per_second=rule.requests_per_second,
                    burst_size=rule.burst_size,
                    rule_id=rule.id
                )
            elif rule_id in self.rate_limiters:
                del self.rate_limiters[rule_id]

        # Save the rules
        self._save_rules()

        return True

    @catch_and_log(component="rate_limiter_registry")
    def delete_rule(self, rule_id: str) -> bool:
        """
        Delete a rate limit rule.

        Args:
            rule_id: ID of the rule to delete

        Returns:
            True if successful, False if rule not found
        """
        if rule_id not in self.rules:
            return False

        # Remove the rule
        del self.rules[rule_id]

        # Remove the rate limiter
        if rule_id in self.rate_limiters:
            del self.rate_limiters[rule_id]

        # Save the rules
        self._save_rules()

        return True

    @catch_and_log(component="rate_limiter_registry")
    def get_rule(self, rule_id: str) -> Optional[RateLimitRule]:
        """
        Get a rate limit rule.

        Args:
            rule_id: ID of the rule to get

        Returns:
            The rule, or None if not found
        """
        return self.rules.get(rule_id)

    @catch_and_log(component="rate_limiter_registry")
    def get_all_rules(self) -> List[RateLimitRule]:
        """
        Get all rate limit rules.

        Returns:
            List of all rules
        """
        return list(self.rules.values())

    @catch_and_log(component="rate_limiter_registry")
    async def check_rate_limit(self, 
                             context: Dict[str, Any],
                             tokens: float = 1.0,
                             wait: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Check if a request is allowed by rate limits.

        Args:
            context: Context information for matching rules
            tokens: Number of tokens to acquire
            wait: Whether to wait for tokens to become available

        Returns:
            Tuple of (allowed, rule_id) - True if allowed, False if limited,
            and the ID of the rule that was applied (or None if no rule matched)
        """
        # Find matching rules
        matching_rules = []
        for rule_id, rule in self.rules.items():
            if rule.is_enabled and self._rule_matches_context(rule, context):
                matching_rules.append(rule_id)

        # No matching rules
        if not matching_rules:
            return True, None

        # Try to acquire from all matching rules
        for rule_id in matching_rules:
            rate_limiter = self.rate_limiters.get(rule_id)
            if rate_limiter:
                allowed = await rate_limiter.acquire(tokens, wait)
                if not allowed:
                    return False, rule_id

        # All rules allowed the request
        return True, matching_rules[0]

    def _rule_matches_context(self, rule: RateLimitRule, context: Dict[str, Any]) -> bool:
        """Check if a rule matches the given context."""
        applies_to = rule.applies_to

        # Empty applies_to means it applies to everything
        if not applies_to:
            return True

        # Check each criterion
        for key, value in applies_to.items():
            if key not in context:
                return False

            # Check if value matches
            if context[key] != value:
                return False

        return True

    def _load_rules(self):
        """Load rules from storage."""
        try:
            rules_file = os.path.join(self.storage_dir, "rules.json")
            if os.path.exists(rules_file):
                with open(rules_file, "r") as f:
                    rules_data = json.load(f)

                # Process each rule
                for rule_data in rules_data:
                    # Convert timestamps
                    if "created_at" in rule_data and isinstance(rule_data["created_at"], str):
                        rule_data["created_at"] = datetime.fromisoformat(rule_data["created_at"])

                    # Create RateLimitRule
                    rule = RateLimitRule(**rule_data)

                    # Store the rule
                    self.rules[rule.id] = rule

                    # Create rate limiter if rule is enabled
                    if rule.is_enabled:
                        self.rate_limiters[rule.id] = TokenBucketRateLimiter(
                            requests_per_second=rule.requests_per_second,
                            burst_size=rule.burst_size,
                            rule_id=rule.id
                        )

                logger.info(f"Loaded {len(self.rules)} rate limit rules")

                # Try to load state
                self._load_state()
        except Exception as e:
            logger.error(f"Error loading rate limit rules: {e}")

    def _save_rules(self):
        """Save rules to storage."""
        try:
            rules_file = os.path.join(self.storage_dir, "rules.json")
            with open(rules_file, "w") as f:
                rules_data = []

                for rule in self.rules.values():
                    # Convert to dict for JSON serialization
                    rule_dict = rule.model_dump()

                    # Convert timestamps to strings
                    if "created_at" in rule_dict and isinstance(rule_dict["created_at"], datetime):
                        rule_dict["created_at"] = rule_dict["created_at"].isoformat()

                    rules_data.append(rule_dict)

                json.dump(rules_data, f, indent=2)

            # Save state
            self._save_state()
        except Exception as e:
            logger.error(f"Error saving rate limit rules: {e}")

    def _load_state(self):
        """Load state for rate limiters."""
        try:
            state_file = os.path.join(self.storage_dir, "state.json")
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state_data = json.load(f)

                # Process each state
                for state_item in state_data:
                    rule_id = state_item.get("rule_id")
                    if rule_id in self.rate_limiters:
                        # Update the rate limiter
                        limiter = self.rate_limiters[rule_id]
                        limiter.tokens = state_item.get("tokens", limiter.burst_size)
                        limiter.last_updated = time.time()  # Always use current time
                        limiter.total_requests = state_item.get("total_requests", 0)
                        limiter.total_limited = state_item.get("total_limited", 0)
        except Exception as e:
            logger.error(f"Error loading rate limiter state: {e}")

    def _save_state(self):
        """Save state for rate limiters."""
        try:
            state_file = os.path.join(self.storage_dir, "state.json")
            with open(state_file, "w") as f:
                state_data = []

                for rule_id, limiter in self.rate_limiters.items():
                    # Get the limiter state
                    state = limiter.get_state()

                    # Convert to dict
                    state_dict = state.model_dump()

                    # Convert timestamps
                    if "last_updated" in state_dict and isinstance(state_dict["last_updated"], datetime):
                        state_dict["last_updated"] = state_dict["last_updated"].isoformat()

                    state_data.append(state_dict)

                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving rate limiter state: {e}")

    def get_all_states(self) -> Dict[str, RateLimitState]:
        """
        Get current state for all rate limiters.

        Returns:
            Dictionary mapping rule IDs to states
        """
        return {rule_id: limiter.get_state() for rule_id, limiter in self.rate_limiters.items()}


def rate_limit(
    requests_per_second: float, 
    burst_size: int = 10, 
    wait: bool = False
):
    """
    Decorator for rate limiting a function.

    Args:
        requests_per_second: Maximum requests per second
        burst_size: Maximum burst size (tokens in the bucket)
        wait: Whether to wait for tokens to become available

    Returns:
        Decorated function
    """
    # Create a rate limiter for this function
    limiter = TokenBucketRateLimiter(
        requests_per_second=requests_per_second,
        burst_size=burst_size
    )

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to acquire a token
            allowed = await limiter.acquire(tokens=1.0, wait=wait)

            if not allowed:
                raise RateLimitExceededError(f"Rate limit exceeded: {requests_per_second} req/s")

            # Call the function
            return await func(*args, **kwargs)

        # Sync version
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Try to acquire a token synchronously (creates event loop if needed)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # Create a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            allowed = loop.run_until_complete(limiter.acquire(tokens=1.0, wait=wait))

            if not allowed:
                raise RateLimitExceededError(f"Rate limit exceeded: {requests_per_second} req/s")

            # Call the function
            return func(*args, **kwargs)

        # Choose the right wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return wrapper
        else:
            return sync_wrapper

    return decorator