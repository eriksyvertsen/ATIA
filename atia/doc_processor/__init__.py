"""
Documentation Processor component.

Parse and extract structured information from API documentation to understand
endpoints, parameters, and authentication requirements.
"""

from atia.doc_processor.processor import DocumentationProcessor, APIEndpoint, APIInfo

__all__ = ["DocumentationProcessor", "APIEndpoint", "APIInfo"]