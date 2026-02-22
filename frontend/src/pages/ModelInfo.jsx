import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer
} from 'recharts'
import { Brain, Layers, ArrowRight, Cpu, Sparkles, Settings } from 'lucide-react'

const lrData = Array.from({ length: 100 }, (_, i) => ({
    epoch: i + 1,
    lr: i < 5
        ? 1e-6 + (1e-4 - 1e-6) * (i / 5)
        : 1e-4 * (0.5 * (1 + Math.cos(Math.PI * (i - 5) / 95))) + 1e-6,
}))

const archBlocks = [
    { label: 'Input\n(Fixed + Moving)', color: '#3b82f6' },
    { label: 'Encoder\n5 blocks', color: '#6366f1' },
    { label: 'Bottleneck\n256 ch', color: '#8b5cf6' },
    { label: 'Decoder\n5 blocks', color: '#a855f7' },
    { label: 'Velocity\nField', color: '#06b6d4' },
    { label: 'Integration\n(S&S ×7)', color: '#10b981' },
    { label: 'Displacement\nField', color: '#f59e0b' },
]

const hyperparams = [
    ['Model', 'VoxelMorphDiff (Diffeomorphic)'],
    ['Parameters', '5,824,131'],
    ['Encoder', '[16, 32, 64, 128, 256]'],
    ['Decoder', '[256, 128, 64, 32, 16]'],
    ['Integration Steps', '7 (Scaling & Squaring)'],
    ['Volume Size', '128 × 128 × 128'],
    ['Voxel Spacing', '1.5 mm isotropic'],
    ['Optimizer', 'AdamW (wd=1e-5)'],
    ['Learning Rate', '1e-4 (Cosine Warmup)'],
    ['Batch Size', '8 (effective 32 w/ accum)'],
    ['AMP', 'BFloat16'],
    ['Loss', 'NCC + Bending Energy + Jacobian'],
    ['GPU', 'NVIDIA H200 (150 GB)'],
    ['Epochs', '100'],
]

const lossWeights = [
    { name: 'NCC (Similarity)', weight: '1.0', desc: 'Normalized cross-correlation' },
    { name: 'Bending Energy', weight: '3.0', desc: 'Smoothness regularization' },
    { name: 'Jacobian Penalty', weight: '0.1', desc: 'Prevents folding' },
    { name: 'Dice (Seg)', weight: '0.5', desc: 'Segmentation alignment' },
]

export default function ModelInfo() {
    return (
        <div className="page-container">
            <div className="page-header">
                <h2>Model Architecture</h2>
                <p>VoxelMorphDiff — Diffeomorphic 3D deformable registration network</p>
            </div>

            {/* Architecture Flow */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header">
                    <div className="card-title">Network Architecture</div>
                    <span className="badge badge-purple"><Brain size={12} /> 5.8M params</span>
                </div>
                <div className="arch-flow">
                    {archBlocks.map((block, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
                            <div className="arch-block" style={{
                                background: `${block.color}12`,
                                borderColor: `${block.color}30`,
                                color: block.color,
                            }}>
                                {block.label.split('\n').map((l, j) => (
                                    <div key={j} style={{ fontSize: j === 0 ? 13 : 11, fontWeight: j === 0 ? 600 : 400, opacity: j === 0 ? 1 : 0.7 }}>{l}</div>
                                ))}
                            </div>
                            {i < archBlocks.length - 1 && <ArrowRight size={14} className="arch-arrow" />}
                        </div>
                    ))}
                </div>
            </div>

            <div className="grid-2">
                {/* Hyperparameters */}
                <div className="card">
                    <div className="card-header">
                        <div className="card-title"><Settings size={15} style={{ marginRight: 6 }} />Hyperparameters</div>
                    </div>
                    {hyperparams.map(([key, val]) => (
                        <div className="metric-row" key={key}>
                            <span className="metric-label">{key}</span>
                            <span className="metric-value" style={{ fontSize: 13 }}>{val}</span>
                        </div>
                    ))}
                </div>

                {/* Loss Weights + LR curve */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                    <div className="card">
                        <div className="card-header">
                            <div className="card-title"><Sparkles size={15} style={{ marginRight: 6 }} />Loss Function</div>
                        </div>
                        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 12 }}>
                            <code style={{ background: 'rgba(255,255,255,0.04)', padding: '4px 8px', borderRadius: 4, fontSize: 12 }}>
                                L = α·NCC + β·Bending + γ·Jacobian
                            </code>
                        </div>
                        {lossWeights.map(l => (
                            <div className="metric-row" key={l.name}>
                                <div>
                                    <span className="metric-label">{l.name}</span>
                                    <div style={{ fontSize: 11, color: '#475569' }}>{l.desc}</div>
                                </div>
                                <span className="badge badge-info">{l.weight}</span>
                            </div>
                        ))}
                    </div>

                    <div className="card">
                        <div className="card-header">
                            <div className="card-title">Learning Rate Schedule</div>
                            <span className="card-subtitle">Cosine w/ 5-epoch warmup</span>
                        </div>
                        <ResponsiveContainer width="100%" height={140}>
                            <AreaChart data={lrData}>
                                <defs>
                                    <linearGradient id="lrGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.3} />
                                        <stop offset="100%" stopColor="#f59e0b" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                                <XAxis dataKey="epoch" stroke="#64748b" fontSize={10} />
                                <YAxis stroke="#64748b" fontSize={10} tickFormatter={v => v.toExponential(0)} />
                                <Tooltip contentStyle={{ background: '#1a1f2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
                                <Area type="monotone" dataKey="lr" stroke="#f59e0b" fill="url(#lrGrad)" strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </div>
    )
}
