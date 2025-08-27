"""HTTP API for Symbiote daemon."""

from typing import Optional
from aiohttp import web
import json
from loguru import logger
from .metrics import get_metrics, LatencyTimer


def create_api_app(daemon) -> web.Application:
    """Create the aiohttp application with routes."""
    app = web.Application()
    app['daemon'] = daemon
    
    # Add routes
    app.router.add_post('/capture', handle_capture)
    app.router.add_get('/context', handle_search)
    app.router.add_get('/suggestions', handle_get_suggestions)
    app.router.add_post('/suggestions/{id}', handle_suggestion_action)
    app.router.add_get('/status', handle_status)
    app.router.add_post('/shutdown', handle_shutdown)
    
    # Additional endpoints from Part 3
    app.router.add_post('/capture/preview', handle_capture_preview)
    app.router.add_post('/suggest', handle_suggest)
    app.router.add_post('/review', handle_review)
    app.router.add_post('/deliberate', handle_deliberate)
    app.router.add_post('/consent', handle_consent)
    app.router.add_get('/receipts/{id}', handle_get_receipt)
    app.router.add_get('/health', handle_health)
    app.router.add_get('/metrics', handle_metrics)
    app.router.add_post('/admin/reindex', handle_reindex)
    app.router.add_post('/admin/config/reload', handle_config_reload)
    
    # Add CORS middleware for local development
    async def cors_middleware(app, handler):
        async def middleware_handler(request):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        return middleware_handler
    
    app.middlewares.append(cors_middleware)
    
    return app


async def handle_capture(request: web.Request) -> web.Response:
    """Handle capture requests."""
    daemon = request.app['daemon']
    metrics = get_metrics()
    
    try:
        with LatencyTimer("capture"):
            data = await request.json()
            
            # Extract parameters
            text = data.get('text')
            if not text:
                return web.json_response(
                    {'error': {'code': 'invalid_request', 'message': 'text is required'}},
                    status=400
                )
            
            # Check text length limit (8k chars)
            if len(text) > 8000:
                return web.json_response(
                    {'error': {'code': 'invalid_request', 'message': 'text too long (max 8000 chars)'}},
                    status=413
                )
            
            capture_type = data.get('type_hint', 'auto')
            if capture_type == 'auto':
                capture_type = 'note'  # Default for now
            
            source = data.get('source', 'text')
            context = data.get('context_hint')
            idempotency_key = data.get('idempotency_key')
            
            # Capture via service
            entry = await daemon.capture_service.capture(
                text=text,
                type=capture_type,
                source=source,
                context=context
            )
            
            metrics.increment_counter("capture.success")
            
            return web.json_response({
                'id': entry.id,
                'status': 'captured',
                'ts': entry.captured_at.isoformat() + 'Z'
            }, status=201)
    
    except Exception as e:
        logger.error(f"Capture error: {e}")
        metrics.increment_counter("capture.error")
        return web.json_response(
            {'error': {'code': 'internal_error', 'message': str(e)}},
            status=500
        )


async def handle_search(request: web.Request) -> web.Response:
    """Handle search/context requests."""
    daemon = request.app['daemon']
    
    try:
        # Get query parameters
        query = request.query.get('q')
        if not query:
            return web.json_response(
                {'error': 'query parameter q is required'},
                status=400
            )
        
        project = request.query.get('project')
        limit = int(request.query.get('limit', 10))
        
        # Execute search
        context_card = await daemon.search_orchestrator.search(
            query=query,
            project_hint=project,
            max_results=limit
        )
        
        # Convert to JSON-serializable format
        return web.json_response({
            'query': context_card.query,
            'results': [
                {
                    'id': r.id,
                    'path': r.path,
                    'title': r.title,
                    'snippet': r.snippet,
                    'score': r.score,
                    'source': r.source.value,
                    'metadata': r.metadata
                }
                for r in context_card.results
            ],
            'quick_actions': context_card.quick_actions,
            'suggestions': context_card.suggestions or [],
            'latency_ms': context_card.latency_ms,
            'strategy_latencies': context_card.strategy_latencies
        })
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return web.json_response(
            {'error': str(e)},
            status=500
        )


async def handle_get_suggestions(request: web.Request) -> web.Response:
    """Get current suggestions."""
    daemon = request.app['daemon']
    
    try:
        # Get suggestions from analytics
        suggestions = await daemon.analytics_indexer.get_link_suggestions()
        
        # Format response
        return web.json_response({
            'suggestions': [
                {
                    'id': f"{s[0]}-{s[1]}",
                    'kind': 'link_suggestion',
                    'text': f"Link {s[0][:8]} to {s[1][:8]}",
                    'score': s[2],
                    'confidence': 'high' if s[2] > 0.8 else 'medium'
                }
                for s in suggestions
            ]
        })
    
    except Exception as e:
        logger.error(f"Suggestions error: {e}")
        return web.json_response(
            {'error': str(e)},
            status=500
        )


async def handle_suggestion_action(request: web.Request) -> web.Response:
    """Handle accept/reject actions on suggestions."""
    daemon = request.app['daemon']
    suggestion_id = request.match_info.get('id')
    
    try:
        data = await request.json()
        action = data.get('action')
        
        if action not in ['accept', 'reject']:
            return web.json_response(
                {'error': 'action must be accept or reject'},
                status=400
            )
        
        # Parse suggestion ID (format: src_id-dst_id)
        parts = suggestion_id.split('-')
        if len(parts) != 2:
            return web.json_response(
                {'error': 'invalid suggestion ID'},
                status=400
            )
        
        src_id, dst_id = parts
        
        # Apply action
        if action == 'accept':
            await daemon.analytics_indexer.promote_link(src_id, dst_id)
        else:
            await daemon.analytics_indexer.reject_link(src_id, dst_id)
        
        daemon.stats['suggestion_count'] += 1
        
        return web.json_response({
            'status': 'ok',
            'action': action,
            'suggestion_id': suggestion_id
        })
    
    except Exception as e:
        logger.error(f"Suggestion action error: {e}")
        return web.json_response(
            {'error': str(e)},
            status=500
        )


async def handle_status(request: web.Request) -> web.Response:
    """Get daemon status."""
    daemon = request.app['daemon']
    
    try:
        status = daemon.get_status()
        return web.json_response(status)
    
    except Exception as e:
        logger.error(f"Status error: {e}")
        return web.json_response(
            {'error': str(e)},
            status=500
        )


async def handle_shutdown(request: web.Request) -> web.Response:
    """Shutdown the daemon."""
    daemon = request.app['daemon']
    
    try:
        logger.info("Shutdown requested via API")
        
        # Schedule shutdown after response
        async def shutdown():
            await asyncio.sleep(0.5)
            await daemon.stop()
            sys.exit(0)
        
        import asyncio
        import sys
        asyncio.create_task(shutdown())
        
        return web.json_response({'status': 'shutting down'})
    
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
        return web.json_response(
            {'error': str(e)},
            status=500
        )


async def handle_capture_preview(request: web.Request) -> web.Response:
    """Handle voice capture preview (partial transcripts)."""
    try:
        data = await request.json()
        # Just acknowledge for now - actual voice processing would be here
        return web.json_response({'status': 'ok'})
    except Exception as e:
        return web.json_response({'error': {'code': 'internal_error', 'message': str(e)}}, status=500)


async def handle_suggest(request: web.Request) -> web.Response:
    """Generate suggestion for current situation."""
    daemon = request.app['daemon']
    metrics = get_metrics()
    
    try:
        from .algorithms import SuggestionGenerator
        from datetime import datetime
        
        data = await request.json()
        situation = data.get('situation', {})
        
        # Get context
        context_items = await daemon.analytics_indexer.get_recent_context(limit=10)
        
        # Generate suggestions
        with LatencyTimer("suggestion"):
            candidates = SuggestionGenerator.generate_heuristic_candidates(
                context_items,
                free_minutes=situation.get('free_minutes', 30),
                project=situation.get('project')
            )
            
            best = SuggestionGenerator.select_best_suggestion(
                candidates,
                query=situation.get('query')
            )
        
        if best:
            # Create receipt
            receipt_id = await daemon.analytics_indexer.create_receipt(
                suggestion_text=best.text,
                sources=best.sources,
                heuristics=best.heuristics,
                confidence=SuggestionGenerator.determine_confidence(best.score)
            )
            
            metrics.record_suggestion_event("generated")
            
            return web.json_response({
                'suggestion': {
                    'text': best.text,
                    'kind': best.kind,
                    'receipts_id': receipt_id
                }
            })
        else:
            return web.json_response({'suggestion': None})
    
    except Exception as e:
        logger.error(f"Suggest error: {e}")
        return web.json_response({'error': {'code': 'internal_error', 'message': str(e)}}, status=500)


async def handle_review(request: web.Request) -> web.Response:
    """Generate daily/weekly synthesis."""
    daemon = request.app['daemon']
    
    try:
        from .algorithms import ThemeSynthesizer
        from datetime import datetime
        
        data = await request.json()
        period = data.get('period', 'daily')
        date_str = data.get('date', datetime.utcnow().strftime('%Y-%m-%d'))
        
        # Get recent items
        recent_items = await daemon.analytics_indexer.get_recent_context(limit=100)
        
        # Extract themes
        themes = ThemeSynthesizer.extract_themes(recent_items)
        
        # Suggest links
        link_suggestions = ThemeSynthesizer.suggest_links(recent_items)
        
        # Create synthesis note
        note_path = daemon.config.vault_path / "insights" / f"{period.title()} Synthesis — {date_str}.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"# {period.title()} Synthesis — {date_str}\n\n"
        
        if themes:
            content += "## Themes\n"
            for theme in themes:
                content += f"- {theme}\n"
            content += "\n"
        
        if link_suggestions:
            content += "## Suggested Links\n"
            for src, dst, score in link_suggestions[:5]:
                content += f"- Link {src[:8]} ↔ {dst[:8]} (confidence: {score:.2f})\n"
                await daemon.analytics_indexer.suggest_link(src, dst, score)
            content += "\n"
        
        with open(note_path, 'w') as f:
            f.write(content)
        
        return web.json_response({'note_path': str(note_path)})
    
    except Exception as e:
        logger.error(f"Review error: {e}")
        return web.json_response({'error': {'code': 'internal_error', 'message': str(e)}}, status=500)


async def handle_deliberate(request: web.Request) -> web.Response:
    """Handle deliberation requests with consent flow."""
    daemon = request.app['daemon']
    
    try:
        from .consent import ConsentManager, ConsentScope
        
        data = await request.json()
        query = data.get('query')
        scope = data.get('scope', {})
        allow_cloud = data.get('allow_cloud', False)
        
        consent_mgr = ConsentManager(daemon.config)
        
        # Collect relevant data
        context_items = await daemon.analytics_indexer.search_by_project(
            scope.get('project')
        ) if scope.get('project') else []
        
        # Prepare preview
        preview = consent_mgr.prepare_preview(
            {"notes": context_items[:10]},
            scope=ConsentScope.ALLOW_BULLETS if allow_cloud else ConsentScope.DENY
        )
        
        # Create action ID
        import ulid
        action_id = f"delib_{ulid.ULID()}"
        
        if preview.token_estimate > 0 and allow_cloud:
            # Store for consent
            consent_mgr.request_consent(action_id, preview)
            
            return web.json_response({
                'plan': ['Collect requirements', 'Analyze options', 'Generate recommendations'],
                'requires_consent': True,
                'redaction_preview': preview.items[:10],  # Limit preview items
                'action_id': action_id,
                'receipts_id': f"rcp_{ulid.ULID()}"
            })
        else:
            # Local-only plan
            return web.json_response({
                'plan': ['Review local notes', 'Apply templates', 'Generate outline'],
                'requires_consent': False,
                'confidence': 'medium',
                'receipts_id': f"rcp_{ulid.ULID()}"
            })
    
    except Exception as e:
        logger.error(f"Deliberate error: {e}")
        return web.json_response({'error': {'code': 'internal_error', 'message': str(e)}}, status=500)


async def handle_consent(request: web.Request) -> web.Response:
    """Handle consent decisions."""
    daemon = request.app['daemon']
    
    try:
        from .consent import ConsentManager
        
        data = await request.json()
        action_id = data.get('action_id')
        allow = data.get('allow', False)
        redactions = data.get('redactions', [])
        
        consent_mgr = ConsentManager(daemon.config)
        final_payload = consent_mgr.apply_consent(action_id, allow, redactions)
        
        if final_payload:
            # Log outbound event
            await daemon.analytics_indexer.log_outbound(
                destination="cloud:example",
                token_count=len(" ".join(final_payload)) // 4,
                categories=["titles", "bullets"],
                redactions={"applied": len(redactions)}
            )
            
            return web.json_response({'status': 'proceeding'})
        else:
            return web.json_response({'status': 'cancelled'})
    
    except Exception as e:
        logger.error(f"Consent error: {e}")
        return web.json_response({'error': {'code': 'internal_error', 'message': str(e)}}, status=500)


async def handle_get_receipt(request: web.Request) -> web.Response:
    """Get receipt by ID."""
    daemon = request.app['daemon']
    receipt_id = request.match_info.get('id')
    
    try:
        # Query from DuckDB
        conn = daemon.analytics_indexer.conn
        result = conn.execute(
            "SELECT * FROM receipts WHERE id = ?",
            [receipt_id]
        ).fetchone()
        
        if result:
            return web.json_response({
                'id': result[0],
                'created_at': str(result[1]),
                'suggestion_text': result[2],
                'sources': json.loads(result[3]) if result[3] else [],
                'heuristics': json.loads(result[4]) if result[4] else [],
                'confidence': result[5],
                'version': result[7]
            })
        else:
            return web.json_response({'error': {'code': 'not_found'}}, status=404)
    
    except Exception as e:
        logger.error(f"Receipt error: {e}")
        return web.json_response({'error': {'code': 'internal_error', 'message': str(e)}}, status=500)


async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    metrics = get_metrics()
    slos = metrics.check_slos()
    
    # Overall health based on SLOs
    healthy = all(slos.values()) if slos else True
    
    return web.json_response({
        'status': 'ok' if healthy else 'degraded',
        'slos': slos,
        'latencies': {
            'capture_p99': metrics.histograms['capture'].get_percentile(99),
            'search_p50': metrics.histograms['search.first_useful'].get_percentile(50)
        }
    })


async def handle_metrics(request: web.Request) -> web.Response:
    """Export metrics."""
    format = request.query.get('format', 'json')
    metrics = get_metrics()
    
    try:
        if format == 'prometheus':
            return web.Response(
                text=metrics.export_metrics('prometheus'),
                content_type='text/plain'
            )
        else:
            return web.Response(
                text=metrics.export_metrics('json'),
                content_type='application/json'
            )
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def handle_reindex(request: web.Request) -> web.Response:
    """Trigger reindexing."""
    daemon = request.app['daemon']
    
    try:
        data = await request.json()
        scope = data.get('scope', 'all')
        
        # Start reindex in background
        import asyncio
        asyncio.create_task(daemon._reindex_all(scope))
        
        return web.json_response({'status': 'reindexing started', 'scope': scope}, status=202)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def handle_config_reload(request: web.Request) -> web.Response:
    """Hot reload configuration."""
    daemon = request.app['daemon']
    
    try:
        from .config import Config
        new_config = Config.load()
        
        # Apply non-structural changes
        daemon.config.privacy = new_config.privacy
        daemon.config.performance = new_config.performance
        daemon.config.synthesis = new_config.synthesis
        
        logger.info("Configuration reloaded")
        return web.json_response({'status': 'reloaded'})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)