import { useState } from 'react';

interface SearchHit {
  rank: number;
  id: string;
  content: string;
  score: number;
  source: string;
}

interface CompareResult {
  query: string;
  top_k: number;
  total_chunks: number;
  elapsed_seconds: number;
  vector: SearchHit[];
  keyword: SearchHit[];
  hybrid: SearchHit[];
}

const MODE_META: Record<string, { label: string; icon: string; color: string }> = {
  vector: { label: '向量检索', icon: '🧠', color: 'blue' },
  keyword: { label: '关键词检索', icon: '🔤', color: 'amber' },
  hybrid: { label: '混合检索 (RRF)', icon: '🔀', color: 'purple' },
};

function ScoreBar({ score, maxScore, color }: { score: number; maxScore: number; color: string }) {
  const pct = maxScore > 0 ? (score / maxScore) * 100 : 0;
  const bgMap: Record<string, string> = {
    blue: 'bg-blue-500',
    amber: 'bg-amber-500',
    purple: 'bg-purple-500',
  };
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${bgMap[color] || 'bg-slate-400'}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <span className="text-xs text-slate-400 w-12 text-right font-mono">
        {score.toFixed(4)}
      </span>
    </div>
  );
}

function ResultColumn({
  mode,
  hits,
  query,
}: {
  mode: string;
  hits: SearchHit[];
  query: string;
}) {
  const meta = MODE_META[mode] || { label: mode, icon: '📋', color: 'slate' };
  const maxScore = hits.length > 0 ? hits[0].score : 1;

  return (
    <div className="flex-1 min-w-0">
      {/* 表头 */}
      <div className={`bg-${meta.color}-50 rounded-t-lg border border-${meta.color}-200 px-3 py-2`}>
        <div className="flex items-center gap-1.5">
          <span>{meta.icon}</span>
          <span className="text-sm font-semibold text-slate-700">{meta.label}</span>
          <span className="text-xs text-slate-400 ml-auto">{hits.length} 条</span>
        </div>
      </div>

      {/* 结果列表 */}
      <div className="border border-t-0 border-slate-200 rounded-b-lg divide-y divide-slate-100">
        {hits.length === 0 && (
          <div className="p-4 text-sm text-slate-400 text-center">无结果</div>
        )}
        {hits.map((hit) => {
          return (
            <div key={`${mode}_${hit.rank}`} className="p-3 hover:bg-slate-50 transition-colors">
              <div className="flex items-center gap-1.5 mb-1">
                <span className="text-xs font-medium text-slate-400">#{hit.rank}</span>
                <ScoreBar score={hit.score} maxScore={maxScore} color={meta.color} />
              </div>
              <p className="text-xs text-slate-600 leading-relaxed line-clamp-3">
                {highlightMatch(hit.content, query)}
              </p>
              <div className="flex items-center gap-1.5 mt-1">
                <span className="text-[10px] text-slate-400">{hit.source}</span>
                <span className="text-[10px] text-slate-300">·</span>
                <span className="text-[10px] text-slate-400 font-mono">ID: {hit.id.slice(0, 8)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** 简单高亮匹配关键词 */
function highlightMatch(text: string, query: string): React.ReactNode {
  if (!query.trim()) return text;
  const words = query.split(/\s+/).filter(Boolean);
  let result = text;
  for (const w of words) {
    if (w.length < 2) continue;
    const escaped = w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    try {
      result = result.replace(
        new RegExp(`(${escaped})`, 'gi'),
        '||HL||$1||/HL||'
      );
    } catch {
      // ignore invalid regex
    }
  }
  const parts = result.split(/\|\|HL\|\||\|\|\/HL\|\|/);
  const highlights: React.ReactNode[] = [];
  let highlighting = false;
  for (const part of parts) {
    if (highlighting) {
      highlights.push(
        <mark key={highlights.length} className="bg-yellow-200 rounded px-0.5">
          {part}
        </mark>
      );
    } else {
      highlights.push(part);
    }
    highlighting = !highlighting;
  }
  return highlights;
}

export default function SearchCompare() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    try {
      const res = await fetch('/api/v1/rag/search_compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: 5 }),
      });
      if (!res.ok) throw new Error('搜索失败');
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  };

  return (
    <div className="space-y-4">
      {/* 搜索栏 */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-xs font-semibold text-slate-500 mb-3 uppercase tracking-wider">
          🔍 检索对比
        </h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入搜索关键词，对比三种检索模式..."
            className="flex-1 border border-slate-300 rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          />
          <button
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            className="px-4 py-2 bg-slate-800 text-white rounded-lg text-sm font-medium hover:bg-slate-700 disabled:bg-slate-300 transition-colors"
          >
            {loading ? '检索中...' : '搜索'}
          </button>
        </div>
      </div>

      {/* 状态栏 */}
      {result && (
        <div className="flex items-center gap-3 text-xs text-slate-500 bg-white rounded-lg border border-slate-200 px-4 py-2">
          <span>📚 知识库: {result.total_chunks} 块</span>
          <span className="text-slate-300">|</span>
          <span>⚡ 耗时: {(result.elapsed_seconds * 1000).toFixed(0)}ms</span>
          <span className="text-slate-300">|</span>
          <span>🔍 查询: "{result.query}"</span>
        </div>
      )}

      {/* 三列对比 */}
      {result && (
        <div className="flex gap-3">
              <ResultColumn mode="vector" hits={result.vector} query={result.query} />
              <ResultColumn mode="keyword" hits={result.keyword} query={result.query} />
              <ResultColumn mode="hybrid" hits={result.hybrid} query={result.query} />
        </div>
      )}

      {/* 结果交叉分析 */}
      {result && (result.vector.length > 0 || result.keyword.length > 0) && (
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <h3 className="text-xs font-semibold text-slate-500 mb-3 uppercase tracking-wider">
            📊 交叉分析
          </h3>
          <div className="grid grid-cols-3 gap-3">
            {['vector', 'keyword', 'hybrid'].map((mode) => {
              const hits = result[mode as keyof typeof result] as SearchHit[];
              const ids = new Set(hits.map((h) => h.id));
              const overlapWithHybrid = result.hybrid.filter((h) => ids.has(h.id)).length;
              const uniqueCount = hits.length - overlapWithHybrid;
              return (
                <div key={mode} className="text-center p-3 bg-slate-50 rounded-lg">
                  <p className="text-xs text-slate-500 mb-1">{MODE_META[mode]?.label || mode}</p>
                  <p className="text-lg font-bold text-slate-700">{hits.length}</p>
                  <p className="text-[10px] text-slate-400">
                    {overlapWithHybrid} 条与混合重叠 · {uniqueCount} 条独有
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 空状态 */}
      {!result && !loading && (
        <div className="text-center py-12 text-sm text-slate-400">
          {query ? '无匹配结果' : '输入关键词开始检索对比'}
        </div>
      )}
    </div>
  );
}
