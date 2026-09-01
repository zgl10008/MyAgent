import { useState, useEffect, useCallback } from 'react';
import { useDropzone } from './useDropzone';
import SearchCompare from '../components/SearchCompare';

interface RagDoc {
  filename: string; mime_type: string; chunk_count: number; char_count: number; uploaded_at: number;
}

const BASE = '/api/v1';

function formatTime(ts: number): string { return new Date(ts * 1000).toLocaleString('zh-CN'); }
function formatSize(chars: number): string {
  if (chars < 1000) return `${chars} 字符`;
  if (chars < 1_000_000) return `${(chars / 1000).toFixed(1)}K 字符`;
  return `${(chars / 1_000_000).toFixed(1)}M 字符`;
}
function isMedical(doc: RagDoc): boolean {
  const kw = ['指南', '血压', '血脂', '体检', '检查', '药品', '说明书', '处方', '诊断', '心电', '超声'];
  return kw.some((k) => doc.filename.includes(k));
}

function FileUploader({ onUploaded }: { onUploaded: () => void }) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const handleFile = async (file: File) => {
    setUploading(true); setError('');
    try {
      const formData = new FormData(); formData.append('file', file);
      const res = await fetch(`${BASE}/rag/ingest`, { method: 'POST', body: formData });
      if (!res.ok) { const err = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(err.detail || `HTTP ${res.status}`); }
      onUploaded();
    } catch (err: any) { setError(err.message); }
    finally { setUploading(false); }
  };
  const { dragOver, rootProps, inputProps } = useDropzone(handleFile);

  return (
    <div {...rootProps} className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${dragOver ? 'border-emerald-500 bg-emerald-50' : 'border-emerald-200 hover:border-emerald-400'}`}>
      <input {...inputProps} />
      <div className="text-3xl mb-2">🏥</div>
      {uploading ? (
        <div>
          <p className="text-sm text-slate-600 mb-2">解析医学文档中...</p>
          <div className="w-full bg-slate-200 rounded-full h-2"><div className="bg-emerald-600 h-2 rounded-full animate-pulse w-2/3" /></div>
        </div>
      ) : (
        <div>
          <p className="text-sm font-medium text-slate-700">拖拽医学文档到此处，或点击上传</p>
          <p className="text-xs text-slate-400 mt-1">支持 PDF、TXT — 检查报告、临床指南、药品说明书</p>
          {error && <p className="text-xs text-red-500 mt-2">{error}</p>}
        </div>
      )}
    </div>
  );
}

export default function DocumentsPage() {
  const [tab, setTab] = useState<'docs' | 'search'>('docs');
  const [docs, setDocs] = useState<RagDoc[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadDocs = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const res = await fetch(`${BASE}/rag/documents`);
      if (!res.ok) throw new Error(res.statusText);
      setDocs(await res.json());
    } catch (err: any) { setError(err.message); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { loadDocs(); }, [loadDocs]);

  const handleDelete = async (filename: string) => {
    if (!confirm(`确定删除 "${filename}"？`)) return;
    try {
      await fetch(`${BASE}/rag/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      await loadDocs();
    } catch (err: any) { alert(err.message); }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="border-b border-emerald-200 px-6 py-4 bg-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-emerald-800">📑 健康档案</h2>
            <p className="text-xs text-slate-400 mt-0.5">上传医学文档，构建专属健康知识库</p>
          </div>
          <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-0.5">
            <button onClick={() => setTab('docs')} className={`px-3 py-1.5 text-xs rounded-md font-medium transition-colors ${tab === 'docs' ? 'bg-white text-emerald-700 shadow-sm' : 'text-slate-500'}`}>📄 档案列表</button>
            <button onClick={() => setTab('search')} className={`px-3 py-1.5 text-xs rounded-md font-medium transition-colors ${tab === 'search' ? 'bg-white text-emerald-700 shadow-sm' : 'text-slate-500'}`}>🔍 检索对比</button>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {tab === 'docs' ? (
          <>
            <FileUploader onUploaded={loadDocs} />
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-3">📋 医学文档列表 ({docs.length})</h3>
              {loading ? <p className="text-sm text-slate-400">加载中...</p> :
               error ? <p className="text-sm text-red-500">加载失败: {error}</p> :
               docs.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  <p className="text-3xl mb-2">📭</p>
                  <p className="text-sm">暂无健康档案，请上传检查报告或医学指南</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead><tr className="border-b border-slate-200 text-left text-slate-500">
                      <th className="pb-2 font-medium">文档名称</th><th className="pb-2 font-medium">类型</th>
                      <th className="pb-2 font-medium">分块</th><th className="pb-2 font-medium">大小</th>
                      <th className="pb-2 font-medium">上传时间</th><th className="pb-2 font-medium">操作</th>
                    </tr></thead>
                    <tbody>
                      {docs.map((doc) => (
                        <tr key={doc.filename} className="border-b border-slate-100">
                          <td className="py-2.5 text-slate-800">
                            {isMedical(doc) && <span className="mr-1">🏥</span>}
                            {doc.filename}
                          </td>
                          <td className="py-2.5 text-slate-500">{doc.mime_type || '-'}</td>
                          <td className="py-2.5 text-slate-500">{doc.chunk_count}</td>
                          <td className="py-2.5 text-slate-500">{formatSize(doc.char_count)}</td>
                          <td className="py-2.5 text-slate-500">{formatTime(doc.uploaded_at)}</td>
                          <td className="py-2.5">
                            <button onClick={() => handleDelete(doc.filename)} className="text-xs text-red-500 hover:text-red-700">删除</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        ) : (
          <SearchCompare />
        )}
      </div>
    </div>
  );
}
