import os
import json
import asyncio
from typing import Dict
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

class OpenAIModelFree(BaseModel):
    agent_name: str = Field(..., description="Name of the agent Renting the property")
    agent_email: str = Field(..., description="Email Address of the agent")
    agent_phone_number: str = Field(
        ..., description="Fee for output token for the OpenAI mode"
    )
    property_address:str = Field(..., description="The Address of the property")
    rent: str = Field(..., description="Monthly rent for the property")
    deposit: str = Field(..., description="Deposit amount for the property")
    admin_fee: str = Field(..., description="Administrative fee for renting the property")
    lease_period: str = Field(..., description="Lease period for the property")
    occupation_date: str = Field(..., description="Occupation date for the property")
    furnished: bool = Field(..., description="Is the property furnished?")
    type_of_property: str = Field(..., description="Type of the property")
    listing_number: str = Field(..., description="Listing number of the property")
    parking: str = Field(..., description="Parking availability for the property")
    parking_cost: str = Field(..., description="Cost of parking for the property")
    pool: bool = Field(..., description="Is there a pool on the property?")
    
    
    
async def extract_structured_data_using_llm(provider: str, api_token: str = None, extra_headers: Dict[str, str] = None):
    print(f"\n--- Extracting Structured Data with {provider} ---")
    
    if api_token is None and provider != "ollama":
        print(f"API token is required for {provider}. Skipping this example.")
        return
    
    browser_config = BrowserConfig(headless=True)
    
    extra_args = {"temperature": 0, "top_p": 0.9, "max_tokens": 2000}
    if extra_headers: 
        extra_args["extra_headers"] = extra_headers
        
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=1,
        page_timeout=80000,
        extraction_strategy=LLMExtractionStrategy(
            provider=provider,
            api_token=api_token,
            schema=OpenAIModelFree.model_json_schema(),
            extraction_type="schema",
            instruction="""From the crawled content, extract all property details and agent details. 
            Do not miss any important details with regards to fees.""",
            extra_args=extra_args
        )
        
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url = "",
            config=crawler_config
        )
        print(result.extracted_content)
    


        
if __name__ == "__main__":
    # Use ollama with llama3.3
    # asyncio.run(
    #     extract_structured_data_using_llm(
    #         provider="ollama/llama3.3", api_token="no-token"
    #     )
    # )
    asyncio.run(
        extract_structured_data_using_llm(
            provider="openai/gpt-4o-mini", api_token=os.getenv("OPENAI_API_KEY")
        )
    )