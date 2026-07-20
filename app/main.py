# -*- coding: utf-8 -*-
"""FastAPI 应用入口：生命周期内初始化异步数据库引擎。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import chat, document, health, sessions
from app.config import get_settings
from app.infrastructure.database.session import configure_session, init_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("启动 {} ({})", settings.app_name, settings.app_env)
    engine = init_engine(settings.database_url)
    configure_session(engine)

    # 自动建表（开发环境；生产应使用 Alembic 迁移）
    try:
        from app.infrastructure.database.models import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("数据库表已就绪")
    except Exception as exc:
        logger.warning("数据库建表失败，可能 PG 未启动: {}", exc)

    app.state.engine = engine
    yield
    await engine.dispose()
    logger.info("关闭 {}", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # CORS — 允许前端开发服务器跨域访问
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health.router, prefix=settings.api_prefix)
    application.include_router(chat.router, prefix=settings.api_prefix)
    application.include_router(document.router, prefix=settings.api_prefix)
    application.include_router(sessions.router, prefix=settings.api_prefix)
    return application


app = create_app()
