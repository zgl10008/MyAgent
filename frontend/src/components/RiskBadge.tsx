type RiskLevel = 'normal' | 'elevated' | 'danger' | 'emergency';

interface Props {
  level: RiskLevel;
  label?: string;
  className?: string;
}

const config: Record<RiskLevel, { bg: string; text: string; border: string; icon: string; defaultLabel: string }> = {
  normal:    { bg: 'bg-green-50',   text: 'text-green-700',   border: 'border-green-200',   icon: '🟢', defaultLabel: '正常' },
  elevated:  { bg: 'bg-amber-50',   text: 'text-amber-700',   border: 'border-amber-200',   icon: '🟡', defaultLabel: '偏高' },
  danger:    { bg: 'bg-red-50',     text: 'text-red-700',     border: 'border-red-200',     icon: '🔴', defaultLabel: '危险' },
  emergency: { bg: 'bg-red-100',    text: 'text-red-800',     border: 'border-red-400',     icon: '🚨', defaultLabel: '紧急' },
};

export default function RiskBadge({ level, label, className = '' }: Props) {
  const c = config[level];
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${c.bg} ${c.text} ${c.border} ${className}`}>
      {c.icon} {label || c.defaultLabel}
    </span>
  );
}

/** 根据血压值自动判定风险等级 */
export function bloodPressureRisk(systolic: number, diastolic: number): RiskLevel {
  if (systolic >= 180 || diastolic >= 110) return 'emergency';
  if (systolic >= 140 || diastolic >= 90) return 'danger';
  if (systolic >= 130 || diastolic >= 85) return 'elevated';
  return 'normal';
}
