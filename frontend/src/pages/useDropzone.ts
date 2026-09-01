import { useState, useRef, useCallback, type InputHTMLAttributes, type Ref } from 'react';

interface DropzoneResult {
  dragOver: boolean;
  rootProps: {
    onDragOver: (e: React.DragEvent) => void;
    onDragLeave: (e: React.DragEvent) => void;
    onDrop: (e: React.DragEvent) => void;
  };
  inputProps: InputHTMLAttributes<HTMLInputElement> & { ref?: Ref<HTMLInputElement> };
}

export function useDropzone(onFile: (file: File) => void): DropzoneResult {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFile(file);
      // reset so same file can be re-selected
      e.target.value = '';
    },
    [onFile]
  );

  return {
    dragOver,
    rootProps: {
      onDragOver: (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(true);
      },
      onDragLeave: () => setDragOver(false),
      onDrop: handleDrop,
    },
    inputProps: {
      ref: inputRef,
      type: 'file',
      accept: '.pdf,.doc,.docx,.txt,.md',
      style: { display: 'none' },
      onChange: handleChange,
    },
  };
}
