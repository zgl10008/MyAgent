import { useState, useEffect, useCallback } from 'react';
import { useDropzone } from './useDropzone';
import SearchCompare from '../components/SearchCompare';

interface RagDoc {
  filename: string;
  mime_type: string;
  chunk_count: number;
  char_count: number;
  uploaded_at: number;
}

const BASE = '/api/v1';

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleString('zh-CN');
}

function formatSize(chars: number): string {
  if (chars < 1000) return `${chars} 字符`;
  if (chars < 1_000_000) return `${(chars / 1000).toFixed(1)}K 字符`;
  return `${(chars / 1_000_000).toFixed(1)}M 字符`;
}

function FileUploader({ onUploaded }: { onUploaded: () => void }) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');

  const handleFile = async (file: File) => {
    setUploading(true);
    setProgress(0);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${BASE}/rag/ingest`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      onUploaded();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  const { dragOver, rootProps, inputProps } = useDropzone(handleFile);

  return (
    <div
      {...rootProps}
      className={`border-2 border-dashed rounded-xl p-6 text-center transition-colors cursor-pointer ${
        dragOver
          ? 'border-blue-500 bg-blue-50'
          : 'border-slate-300 hover:border-slate-400'
      }`}
    >
      <input {...inputProps} />
      <div className="text-3xl mb-2">📄</div>
      {uploading ? (
        <div>
          <p className="text-sm text-slate-600 mb-2">解析中...</p>
          <div className="w-full bg-slate-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      ) : (
        <div>
          <p className="text-sm font-medium text-slate-700">
            拖拽文件到此处，或点击选择文件
          </p>
          <p className="text-xs text-slate-400 mt-1">支持 TXT、PDF、Markdown</p>
          {error && <p className="text-xs text-red-500 mt-2">{error}</p>}
        </div>
      )}
    </div>
  );
}

export default function DocumentsPage() {
  const [tab, setTab] = useState<'docs' | 'search'>('docs');
  const [docs, setDocs] = useState<RagDoc[]>([]);
  const [searchRefreshKey, setSearchRefreshKey] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadDocs = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${BASE}/rag/documents`);
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      setDocs(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDocs();
  }, [loadDocs]);

  const handleDelete = async (filename: string) => {
    if (!confirm(`确定删除 "${filename}"？`)) return;
    try {
      const res = await fetch(
        `${BASE}/rag/documents/${encodeURIComponent(filename)}`,
        { method: 'DELETE' }
      );
      if (!res.ok) throw new Error('删除失败');
      await loadDocs();
    } catch (err: any) {
      alert(err.message);
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="border-b border-slate-200 px-6 py-3 bg-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-800">📄 文档管理</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              上传文档，构建 RAG 知识库
            </p>
          </div>
          <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-0.5">
            <button
              onClick={() => setTab('docs')}
              className={`px-3 py-1.5 text-xs rounded-md font-medium transition-colors ${
                tab === 'docs'
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              📄 文档
            </button>
            <button
              onClick={() => setTab('search')}
              className={`px-3 py-1.5 text-xs rounded-md font-medium transition-colors ${
                tab === 'search'
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              🔍 检索对比
            </button>
          </div>
        </div>
      </div>
      <div className="p-6 space-y-6">
        {tab === 'docs' ? (
          <>
            <FileUploader onUploaded={loadDocs} />

            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-3">
                知识库文档 ({docs.length})
              </h3>
              {loading ? (
                <div className="text-sm text-slate-400">加载中...</div>
              ) : error ? (
                <div className="text-sm text-red-500">加载失败: {error}</div>
              ) : docs.length === 0 ? (
                <div className="text-sm text-slate-400">暂无文档，请上传</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 text-left text-slate-500">
                        <th className="pb-2 font-medium">文件名</th>
                        <th className="pb-2 font-medium">类型</th>
                        <th className="pb-2 font-medium">分块数</th>
                        <th className="pb-2 font-medium">大小</th>
                        <th className="pb-2 font-medium">上传时间</th>
                        <th className="pb-2 font-medium">操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {docs.map((doc) => (
                        <tr key={doc.filename} className="border-b border-slate-100">
                          <td className="py-2.5 text-slate-800">{doc.filename}</td>
                          <td className="py-2.5 text-slate-500">{doc.mime_type || '-'}</td>
                          <td className="py-2.5 text-slate-500">{doc.chunk_count}</td>
                          <td className="py-2.5 text-slate-500">{formatSize(doc.char_count)}</td>
                          <td className="py-2.5 text-slate-500">{formatTime(doc.uploaded_at)}</td>
                          <td className="py-2.5">
                            <button
                              onClick={() => handleDelete(doc.filename)}
                              className="text-xs text-red-500 hover:text-red-700 transition-colors"
                            >
                              删除
                            </button>
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
          <SearchCompare refreshKey={searchRefreshKey} />
        )}
      </div>
    </div>
  );
}
