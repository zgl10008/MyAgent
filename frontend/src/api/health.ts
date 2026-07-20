import type { HealthResponse, ReadinessResponse } from '../types';

const BASE = '/api/v1';

/** 健康检查 */
export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}

/** 就绪检查 */
export async function checkReadiness(): Promise<ReadinessResponse> {
  const res = await fetch(`${BASE}/health/ready`);
  return res.json();
}
