# 001: Use OpenAI Chat Completions API for primary public API interface

*   **Status:** Proposed
*   **Date:** 2025-05-02

## Context

One of the stated goals of the API layer is to normalize API access across models and backend providers. This requires a choice of API that will be stable and full featured enough to be useful for our partners.

This will help:
- prevent vendor lock-in by making it easy to switch vendors without changing the API
- allow users to test different models without the need to understand the nuances of each model and vendor's api

Options considered:
- Create a bespoke API for our specific needs. This would require re-inventing the wheel and will place a burden on users to learn and write tooling around our API.
- Use an existing API. We are already using Bedrock and VertexAI, both of which offer fine APIs. Neither have the momentum of OpenAI's in the community.

## Decision

We will use OpenAI's Chat Completions API and embedding API as the primary interface for our API service. 

## Consequences

*   **Positive:**
    *   Leverages existing knowledge, tools, and documentation.
    *   OpenAI Chat Completions is a defacto standard in this space
    *   Our existing chat bot already uses it
    *   Allows other tools to just plug-in to the api.
*   **Negative:**
    *   Even though OpenAI's API is the de facto standard, it is not a real standard with 3rd party governance. 
    *   We are tied to decisions made by OpenAI that may not align with our goals
    *   The API does not allow access to all the features of the different vendors. We will need to address features that are unique to other models if users demand them. 
*   **Neutral:**
    *   Regardless of the interface chosen, we will need to maintain a translation layer that converts from the public interface to vendor/model API's. This will require staffing people with the skills to monitor changes across diverse offerings and make changes to accommodate a quickly shifting tech landscape. 
*   **Future:**
    *   This choice may need to be reconsidered if technology moves in a direction that can not be easily supported by the OpenAI API.
    *   If we need to support features outside this API, we may be able to create backward compatible extensions to the API to suit our needs.

