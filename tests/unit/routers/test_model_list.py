
def test_model_list(client, mock_llm_models):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data == [m.model_dump() for m in mock_llm_models]