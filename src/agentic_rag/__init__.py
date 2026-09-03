"""Advanced Agentic RAG Platform."""

import asyncio
import sys

if sys.platform == "win32":
    # asyncpg's connection teardown is incompatible with Windows' default
    # ProactorEventLoop (raises RuntimeError/AttributeError on close). This
    # must run before any event loop is created, so it lives at package import
    # time rather than in application startup code.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
