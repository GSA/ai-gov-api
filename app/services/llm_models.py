import json
import structlog
import aioboto3
from botocore.config import Config

from app.schema.bedrock import ConverseRequest, ConverseResponse, CohereRequest
from app.services.billing import billing_queue
from app.config.settings import settings

log = structlog.get_logger()

retry_config = Config(
    retries={
        "max_attempts": 5, 
        "mode": "standard", 
    },
    region_name=settings.aws_default_region
)

async def invoke_converse_model(payload: ConverseRequest) -> ConverseResponse:
    session = aioboto3.Session()
    async with session.client("bedrock-runtime", config=retry_config) as client:
        body = payload.dict(exclude_none=True)

        arn = getattr(settings.bedrock_models, payload.model_id).arn
        body['modelId'] = arn
        response = await client.converse(**body)
        await billing_queue.put(response['usage'])
        log.info("bedrock metrics", model=payload.model_id, **response['metrics'])
      
        return ConverseResponse(**response)

async def get_chat(payload: ConverseRequest):
    session = aioboto3.Session()

    async with session.client("bedrock-runtime", config=retry_config) as client:
        body = payload.dict(exclude_none=True)
        arn = settings.bedrock_models[payload.model_id].arn
        body['modelId'] = arn

        stream = await client.converse_stream(**body)
        
        async for chunk in stream['stream']:
            if 'contentBlockDelta' in chunk:
                yield f"data: {chunk['contentBlockDelta']['delta']['text']}\n\n"
            elif 'metadata' in chunk:
                await billing_queue.put(chunk['metadata']['usage'])
                log.info(**chunk['metadata']['metrics'])

        

async def get_embeddings(payload: CohereRequest):
    session = aioboto3.Session()
    async with session.client("bedrock-runtime", config=retry_config) as client:
        body = payload.model_dump_json(exclude_none=True)
        modelId = settings.cohere_embed_model_id 
        
        response = await client.invoke_model(
            body=body,
            modelId=modelId,
            accept = '*/*',
            contentType = 'application/json'
            )
        
        headers = response['ResponseMetadata']['HTTPHeaders']
        latency = headers['x-amzn-bedrock-invocation-latency']
        log.info("embedding", latency=latency, model=modelId)
        input_tokens = headers['x-amzn-bedrock-input-token-count']

        await billing_queue.put({"model": modelId, "input_tokens:": input_tokens})
        response_body = json.loads(await response.get("body").read())

        return response_body
