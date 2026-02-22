import { useState } from 'react'
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, ScatterChart, Scatter, ZAxis
} from 'recharts'
import { Eye, Layers, Grid3X3, Maximize } from 'lucide-react'

const displacementData = Array.from({ length: 20 }, (_, i) => ({
    region: `R${i + 1}`,
    magnitude: Math.random() * 8 + 1,
    color: `hsl(${200 + Math.random() * 60}, 70%, 50%)`,
}))

const jacobianData = Array.from({ length: 50 }, () => ({
    x: Math.random() * 100,
    y: Math.random() * 100,
    value: 0.7 + Math.random() * 0.6,
}))

const metrics = {
    nccBefore: 0.32, nccAfter: 0.91,
    ssimBefore: 0.45, ssimAfter: 0.88,
    mseBefore: 0.082, mseAfter: 0.012,
    dispMean: '3.2 mm', dispMax: '12.8 mm',
    jacMean: 1.02, jacNeg: '0.3%',
}

export default function ResultsViewer() {
    const [activeTab, setActiveTab] = useState('comparison')

    return (
        <div className="page-container">
            <div className="page-header">
                <h2>Results Viewer</h2>
                <p>Inspect registration outputs, deformation fields, and quality metrics</p>
            </div>

            {/* Tabs */}
            <div className="tabs">
                {[
                    { id: 'comparison', icon: <Eye size={14} />, label: 'Comparison' },
                    { id: 'deformation', icon: <Grid3X3 size={14} />, label: 'Deformation' },
                    { id: 'jacobian', icon: <Layers size={14} />, label: 'Jacobian' },
                    { id: 'metrics', icon: <Maximize size={14} />, label: 'Metrics' },
                ].map(tab => (
                    <button
                        key={tab.id}
                        className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            {tab.icon} {tab.label}
                        </span>
                    </button>
                ))}
            </div>

            {activeTab === 'comparison' && (
                <>
                    <div className="slice-viewer-grid" style={{ marginBottom: 24 }}>
                        {['Fixed (Target)', 'Moving (Source)', 'Warped (Aligned)', 'Difference'].map((label) => (
                            <div key={label} className="slice-viewer" style={{ background: '#0a0e1a', minHeight: 220 }}>
                                <div style={{
                                    width: '100%', height: '100%',
                                    background: `radial-gradient(ellipse at ${40 + Math.random() * 20}% ${40 + Math.random() * 20}%, rgba(100,150,200,0.3) 0%, rgba(30,50,80,0.2) 40%, rgba(10,14,26,1) 70%)`,
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                }}>
                                    <span className="slice-label">{label}</span>
                                    <span style={{ fontSize: 11, color: '#475569' }}>CT Slice — Axial View</span>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="card">
                        <div className="card-title" style={{ marginBottom: 12 }}>Slice Navigator</div>
                        <input type="range" min={0} max={127} defaultValue={64}
                            style={{ width: '100%', accentColor: '#3b82f6' }}
                        />
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#64748b', marginTop: 4 }}>
                            <span>Slice 0</span><span>Slice 64</span><span>Slice 127</span>
                        </div>
                    </div>
                </>
            )}

            {activeTab === 'deformation' && (
                <div className="card">
                    <div className="card-header">
                        <div className="card-title">Displacement Magnitude by Region</div>
                        <span className="badge badge-info">mm</span>
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={displacementData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                            <XAxis dataKey="region" stroke="#64748b" fontSize={11} />
                            <YAxis stroke="#64748b" fontSize={11} unit=" mm" />
                            <Tooltip contentStyle={{ background: '#1a1f2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
                            <Bar dataKey="magnitude" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            )}

            {activeTab === 'jacobian' && (
                <div className="card">
                    <div className="card-header">
                        <div className="card-title">Jacobian Determinant Distribution</div>
                        <div style={{ display: 'flex', gap: 12 }}>
                            <span style={{ fontSize: 11, color: '#3b82f6' }}>● Expansion (&gt;1)</span>
                            <span style={{ fontSize: 11, color: '#ef4444' }}>● Contraction (&lt;1)</span>
                        </div>
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                        <ScatterChart>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                            <XAxis dataKey="x" stroke="#64748b" fontSize={11} name="X" />
                            <YAxis dataKey="y" stroke="#64748b" fontSize={11} name="Y" />
                            <ZAxis dataKey="value" range={[20, 100]} />
                            <Tooltip contentStyle={{ background: '#1a1f2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
                            <Scatter data={jacobianData.filter(d => d.value >= 1)} fill="#3b82f6" fillOpacity={0.6} />
                            <Scatter data={jacobianData.filter(d => d.value < 1)} fill="#ef4444" fillOpacity={0.6} />
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>
            )}

            {activeTab === 'metrics' && (
                <div className="grid-2">
                    <div className="card">
                        <div className="card-title" style={{ marginBottom: 16 }}>Image Similarity</div>
                        <MetricRow label="NCC (Before)" value={metrics.nccBefore.toFixed(2)} />
                        <MetricRow label="NCC (After)" value={metrics.nccAfter.toFixed(2)} highlight />
                        <MetricRow label="SSIM (Before)" value={metrics.ssimBefore.toFixed(2)} />
                        <MetricRow label="SSIM (After)" value={metrics.ssimAfter.toFixed(2)} highlight />
                        <MetricRow label="MSE (Before)" value={metrics.mseBefore.toFixed(3)} />
                        <MetricRow label="MSE (After)" value={metrics.mseAfter.toFixed(3)} highlight />
                    </div>
                    <div className="card">
                        <div className="card-title" style={{ marginBottom: 16 }}>Deformation Quality</div>
                        <MetricRow label="Mean Displacement" value={metrics.dispMean} />
                        <MetricRow label="Max Displacement" value={metrics.dispMax} />
                        <MetricRow label="Jacobian Mean" value={metrics.jacMean.toFixed(3)} />
                        <MetricRow label="Negative Jacobian %" value={metrics.jacNeg} highlight />
                        <MetricRow label="Diffeomorphic" value="✓ Yes" highlight />
                        <MetricRow label="Topology Preserved" value="✓ Yes" highlight />
                    </div>
                </div>
            )}
        </div>
    )
}

function MetricRow({ label, value, highlight }) {
    return (
        <div className="metric-row">
            <span className="metric-label">{label}</span>
            <span className="metric-value" style={highlight ? { color: '#10b981' } : {}}>{value}</span>
        </div>
    )
}
