import type { DocumentInfo, DocumentUploadResponse } from '../types';

const BASE = '/api/v1';

/** 获取文档列表 */
export async function listDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${BASE}/documents`);
  if (!res.ok) throw new Error(`获取文档列表失败: ${res.statusText}`);
  return res.json();
}

/** 上传文档 */
export async function uploadDocument(
  file: File,
  onProgress?: (pct: number) => void
): Promise<DocumentUploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${BASE}/documents/upload`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail || `HTTP ${xhr.status}`));
        } catch {
          reject(new Error(`上传失败: HTTP ${xhr.status}`));
        }
      }
    };

    xhr.onerror = () => reject(new Error('网络错误'));
    const formData = new FormData();
    formData.append('file', file);
    xhr.send(formData);
  });
}
