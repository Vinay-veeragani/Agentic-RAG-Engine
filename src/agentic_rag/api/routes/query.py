from fastapi import APIRouter

from agentic_rag.agents.planner import QueryDecomposer, RetrievalPlanner
from agentic_rag.agents.query_analyzer import QueryAnalyzer, QueryExpander
from agentic_rag.api.dependencies.llm import LLMProviderDep
from agentic_rag.api.schemas.query import QueryAnalyzeRequest, QueryAnalyzeResponse
from agentic_rag.core.config import get_settings

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/analyze", response_model=QueryAnalyzeResponse)
async def analyze_query(
    body: QueryAnalyzeRequest, llm: LLMProviderDep
) -> QueryAnalyzeResponse:
    settings = get_settings()

    analysis = await QueryAnalyzer(llm).analyze(body.query)
    plan = await RetrievalPlanner(
        llm, max_iterations_ceiling=settings.max_retrieval_iterations
    ).plan(body.query, analysis)

    expanded_queries = None
    if plan.expand_query:
        expanded_queries = (await QueryExpander(llm).expand(body.query)).expanded_queries

    subqueries = None
    if plan.decompose:
        subqueries = (await QueryDecomposer(llm).decompose(body.query)).subqueries

    return QueryAnalyzeResponse(
        query=body.query,
        analysis=analysis,
        plan=plan,
        expanded_queries=expanded_queries,
        subqueries=subqueries,
    )
