#!/usr/bin/env python3
"""
Consolidated Site24x7 Reporter with Full Integration
Handles metrics, logs, and system monitoring for Docker Agent
"""

import asyncio
import time
import logging
import psutil
import sqlite3
from typing import Dict, List, Optional
import aiohttp
from dataclasses import dataclass
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MetricData:
    metric_name: str
    value: float
    count: int
    time_stamp: int
    tags: Dict[str, str]

class ConsolidatedSite24x7Reporter:
    def __init__(self):
        # Site24x7 configuration
        self.api_key = os.getenv('SITE24X7_API_KEY', 'in_24be7e829d6ca9b6dd72ca278c32e2bf')
        self.app_key = os.getenv('SITE24X7_APP_KEY', 'e5b0f39bd1c6a990b6ca6ef78104bff7')
        self.license_key = os.getenv('SITE24X7_LICENSE_KEY', self.api_key)
        
        # Endpoints
        self.metrics_endpoint = f"https://plusinsight.site24x7.in/metrics/v2/data?app.key={self.app_key}&license.key={self.license_key}"
        self.logs_endpoint = f"https://logc.site24x7.com/event/receiver?token={self.api_key}"
        
        # Data sources
        self.db_path = "/app/metrics.db"
        self.prometheus_url = "http://host.docker.internal:8001/metrics"
        
        self.running = False
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Start the reporter with all monitoring capabilities"""
        self.running = True
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'Content-Type': 'application/json'}
        )
        logger.info("✅ Consolidated Site24x7 Reporter started")
        await self._reporting_loop()

    async def stop(self):
        """Stop the reporter gracefully"""
        self.running = False
        if self.session:
            await self.session.close()
        logger.info("🛑 Consolidated Site24x7 Reporter stopped")

    async def _reporting_loop(self):
        """Main reporting loop - runs every 30 seconds"""
        while self.running:
            try:
                await asyncio.gather(
                    self._collect_and_send_app_logs(),
                    self._collect_and_send_system_metrics(),
                    self._collect_and_send_prometheus_metrics(),
                    return_exceptions=True
                )
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"❌ Reporting loop error: {e}")
                await asyncio.sleep(60)

    async def _collect_and_send_app_logs(self):
        """Collect application logs from SQLite and send to Site24x7"""
        try:
            if not os.path.exists(self.db_path):
                logger.debug(f"Database not found: {self.db_path}")
                return

            logs = self._get_recent_logs_from_db()
            if logs:
                await self._send_logs_to_site24x7(logs)
                logger.info(f"📊 Sent {len(logs)} app logs to Site24x7")
        except Exception as e:
            logger.error(f"❌ Failed to collect app logs: {e}")

    def _get_recent_logs_from_db(self, minutes: int = 5) -> List[Dict]:
        """Get recent logs from SQLite database"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 50
                """, (cutoff_time.isoformat(),))
                
                rows = cursor.fetchall()
                logs = []
                
                for row in rows:
                    log_entry = {
                        "_zl_timestamp": int(time.time() * 1000),
                        "trace_id": row['trace_id'],
                        "framework": row['framework'],
                        "model": row['model'],
                        "vector_store": row['vector_store'],
                        "input_tokens": row['input_tokens'],
                        "output_tokens": row['output_tokens'],
                        "total_tokens": row['total_tokens'],
                        "latency_ms": row['latency_ms'],
                        "total_cost": row['total_cost'],
                        "status": row['status'],
                        "service": "docker-agent",
                        "log_level": "INFO" if row['status'] == 'completed' else "ERROR"
                    }
                    
                    if row.get('error_message'):
                        log_entry['error_message'] = row['error_message']
                    
                    logs.append(log_entry)
                
                return logs
        except Exception as e:
            logger.error(f"❌ Failed to read from database: {e}")
            return []

    async def _send_logs_to_site24x7(self, logs: List[Dict]):
        """Send logs to Site24x7 AppLogs"""
        try:
            for log in logs:
                async with self.session.post(
                    self.logs_endpoint,
                    json=log
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.warning(f"⚠️ Log send failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"❌ Error sending logs to Site24x7: {e}")

    async def _collect_and_send_system_metrics(self):
        """Collect and send system metrics"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            current_timestamp = int(time.time() * 1000)
            
            metrics = [
                {
                    "metric_name": "docker_agent_system_cpu_percent",
                    "value": cpu_percent,
                    "count": 1,
                    "time_stamp": current_timestamp,
                    "tags": {"service": "docker-agent", "type": "system"}
                },
                {
                    "metric_name": "docker_agent_system_memory_percent",
                    "value": memory.percent,
                    "count": 1,
                    "time_stamp": current_timestamp,
                    "tags": {"service": "docker-agent", "type": "system"}
                },
                {
                    "metric_name": "docker_agent_system_disk_percent",
                    "value": disk.percent,
                    "count": 1,
                    "time_stamp": current_timestamp,
                    "tags": {"service": "docker-agent", "type": "system"}
                }
            ]
            
            await self._send_metrics_to_site24x7(metrics)
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")

    async def _collect_and_send_prometheus_metrics(self):
        """Collect metrics from Prometheus and send to Site24x7"""
        try:
            async with self.session.get(self.prometheus_url) as response:
                if response.status != 200:
                    logger.debug(f"Prometheus not available: {response.status}")
                    return
                
                raw_metrics = await response.text()
                parsed_metrics = self._parse_prometheus_metrics(raw_metrics)
                
                if parsed_metrics:
                    await self._send_metrics_to_site24x7(parsed_metrics)
                    logger.debug(f"📈 Sent {len(parsed_metrics)} Prometheus metrics")
        except Exception as e:
            logger.debug(f"Prometheus metrics collection failed: {e}")

    def _parse_prometheus_metrics(self, raw: str) -> List[Dict]:
        """Parse Prometheus metrics format"""
        result = []
        current_timestamp = int(time.time() * 1000)
        
        for line in raw.splitlines():
            if line.startswith("#") or line.strip() == "":
                continue
            if "docker_agent_" not in line:
                continue
            
            try:
                if "{" in line:
                    metric_part, value_str = line.split("} ")
                    tags_str = metric_part.split("{")[1]
                    metric_name = metric_part.split("{")[0]
                    value = float(value_str.strip())
                    
                    tags = {}
                    for tag in tags_str.split(","):
                        if "=" in tag:
                            k, v = tag.split("=", 1)
                            tags[k] = v.strip('"')
                else:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        metric_name, value = parts[0], float(parts[1])
                        tags = {}
                
                result.append({
                    "metric_name": metric_name,
                    "value": value,
                    "count": 1,
                    "time_stamp": current_timestamp,
                    "tags": {**tags, "source": "prometheus"}
                })
            except Exception as e:
                logger.debug(f"Skipping malformed metric line: {line}")
        
        return result

    async def _send_metrics_to_site24x7(self, metrics: List[Dict]):
        """Send metrics to Site24x7"""
        try:
            async with self.session.post(
                self.metrics_endpoint,
                json=metrics
            ) as response:
                if response.status == 200:
                    logger.debug("✅ Metrics sent to Site24x7")
                else:
                    error_text = await response.text()
                    logger.warning(f"⚠️ Metrics send failed: {response.status}")
        except Exception as e:
            logger.error(f"❌ HTTP error sending metrics: {e}")

async def main():
    """Main entry point"""
    reporter = ConsolidatedSite24x7Reporter()
    try:
        await reporter.start()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    finally:
        await reporter.stop()

if __name__ == "__main__":
    asyncio.run(main())
