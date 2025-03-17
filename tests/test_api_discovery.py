import pytest
from unittest.mock import patch

from atia.api_discovery.discovery import APIDiscovery, APICandidate


def test_api_discovery_init():
    """Test that APIDiscovery initializes correctly."""
    discovery = APIDiscovery()
    # Just check that initialization doesn't raise an exception
    assert discovery is not None


@patch("atia.api_discovery.discovery.get_completion")
def test_formulate_search_query(mock_get_completion):
    """Test that a search query is formulated correctly."""
    mock_get_completion.return_value = "translation API language services"

    discovery = APIDiscovery()
    query = discovery.formulate_search_query("Translate text from English to French")

    assert query == "translation API language services"
    mock_get_completion.assert_called_once()


@pytest.mark.asyncio
async def test_execute_web_search_dummy():
    """Test that dummy search results are returned when no API key is provided."""
    discovery = APIDiscovery()
    discovery.rapid_api_key = None  # Ensure we use dummy results

    results = await discovery.execute_web_search("translation API")

    assert len(results) > 0
    assert "name" in results[0]
    assert "provider" in results[0]
    assert "documentation_url" in results[0]


@pytest.mark.asyncio
@patch("atia.api_discovery.discovery.get_embedding")
async def test_rank_by_relevance(mock_get_embedding):
    """Test that search results are ranked by relevance."""
    # Mock embeddings for testing
    mock_get_embedding.side_effect = [
        [0.1, 0.2, 0.3],  # Capability embedding
        [0.1, 0.2, 0.3],  # Perfect match (cosine = 1.0)
        [0.2, 0.2, 0.3],  # Good match (cosine < 1.0)
        [-0.1, -0.2, -0.3]  # Poor match (cosine < 0)
    ]

    discovery = APIDiscovery()

    # Create test search results
    search_results = [
        {
            "name": "Perfect API",
            "provider": "Provider A",
            "description": "Perfect match",
            "documentation_url": "https://example.com/perfect"
        },
        {
            "name": "Good API",
            "provider": "Provider B",
            "description": "Good match",
            "documentation_url": "https://example.com/good"
        },
        {
            "name": "Poor API",
            "provider": "Provider C",
            "description": "Poor match",
            "documentation_url": "https://example.com/poor"
        }
    ]

    ranked_results = await discovery.rank_by_relevance(search_results, "Translate text")

    assert len(ranked_results) == 3
    assert ranked_results[0].name == "Perfect API"
    assert ranked_results[1].name == "Good API"
    assert ranked_results[2].name == "Poor API"
    assert ranked_results[0].relevance_score > ranked_results[1].relevance_score > ranked_results[2].relevance_score


@pytest.mark.asyncio
@patch("atia.api_discovery.discovery.APIDiscovery.formulate_search_query")
@patch("atia.api_discovery.discovery.APIDiscovery.execute_web_search")
@patch("atia.api_discovery.discovery.APIDiscovery.rank_by_relevance")
async def test_search_for_api(mock_rank, mock_execute, mock_formulate):
    """Test the full API search pipeline."""
    mock_formulate.return_value = "test query"
    mock_execute.return_value = [{"name": "Test API"}]

    # Create mock ranked results
    mock_candidates = [
        APICandidate(
            name="API 1",
            provider="Provider 1",
            description="Description 1",
            documentation_url="https://example.com/1",
            relevance_score=0.9
        ),
        APICandidate(
            name="API 2",
            provider="Provider 2",
            description="Description 2",
            documentation_url="https://example.com/2",
            relevance_score=0.8
        )
    ]
    mock_rank.return_value = mock_candidates

    discovery = APIDiscovery()
    results = await discovery.search_for_api("Translate text", num_results=1)

    assert len(results) == 1
    assert results[0].name == "API 1"
    assert results[0].relevance_score == 0.9

    mock_formulate.assert_called_once()
    mock_execute.assert_called_once()
    mock_rank.assert_called_once()
    