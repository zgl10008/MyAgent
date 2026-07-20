import type { ChatMessage } from '../types';

const BASE = '/api/v1';

export interface SessionMeta {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
}

export interface SessionDetail {
  id: string;
  title: string;
  messages: ChatMessage[];
}

/** 获取会话列表 */
export async function listSessions(): Promise<SessionMeta[]> {
  const res = await fetch(`${BASE}/sessions`);
  if (!res.ok) throw new Error(`获取会话列表失败: ${res.statusText}`);
  return res.json();
}

/** 获取会话详情（含历史消息） */
export async function getSession(id: string): Promise<SessionDetail> {
  const res = await fetch(`${BASE}/sessions/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`获取会话失败: ${res.statusText}`);
  return res.json();
}

/** 删除会话 */
export async function deleteSession(id: string): Promise<void> {
  const res = await fetch(`${BASE}/sessions/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`删除会话失败: ${res.statusText}`);
}
