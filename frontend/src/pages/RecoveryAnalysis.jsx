import { useState } from 'react'
import {
    LineChart, Line, AreaChart, Area, RadarChart, Radar,
    PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'
import { Activity, TrendingUp, Heart, AlertTriangle } from 'lucide-react'

const recoveryTimeline = [
    { timepoint: 'Baseline', score: 0, upper: 0, middle: 0, lower: 0 },
    { timepoint: '1 Month', score: 0.25, upper: 0.30, middle: 0.22, lower: 0.18 },
    { timepoint: '3 Months', score: 0.52, upper: 0.62, middle: 0.50, lower: 0.38 },
    { timepoint: '6 Months', score: 0.74, upper: 0.82, middle: 0.72, lower: 0.61 },
    { timepoint: '12 Months', score: 0.89, upper: 0.95, middle: 0.88, lower: 0.79 },
]

const radarData = [
    { metric: 'Upper Lobe', A: 0.95, fullMark: 1 },
    { metric: 'Middle Lobe', A: 0.88, fullMark: 1 },
    { metric: 'Lower Lobe', A: 0.79, fullMark: 1 },
    { metric: 'Tissue Density', A: 0.85, fullMark: 1 },
    { metric: 'Volume Change', A: 0.92, fullMark: 1 },
    { metric: 'Deformation', A: 0.87, fullMark: 1 },
]

const patients = [
    { id: 'PAT-001', trend: 'Improving', score: 0.89, rate: '+0.18/month', status: 'Complete' },
    { id: 'PAT-002', trend: 'Improving', score: 0.74, rate: '+0.12/month', status: 'Partial' },
    { id: 'PAT-003', trend: 'Stable', score: 0.52, rate: '+0.04/month', status: 'Early' },
    { id: 'PAT-004', trend: 'Improving', score: 0.91, rate: '+0.20/month', status: 'Complete' },
    { id: 'PAT-005', trend: 'Declining', score: 0.38, rate: '-0.02/month', status: 'Concern' },
]

export default function RecoveryAnalysis() {
    const [selectedPatient, setSelectedPatient] = useState('PAT-001')

    return (
        <div className="page-container">
            <div className="page-header">
                <h2>Recovery Analysis</h2>
                <p>Track longitudinal lung recovery across timepoints</p>
            </div>

            {/* Stats */}
            <div className="stats-grid">
                <StatMini icon={<Heart size={18} />} label="Avg Recovery" value="72%" color="#10b981" />
                <StatMini icon={<TrendingUp size={18} />} label="Improving" value="4 / 5" color="#3b82f6" />
                <StatMini icon={<Activity size={18} />} label="Avg Rate" value="+0.13/mo" color="#8b5cf6" />
                <StatMini icon={<AlertTriangle size={18} />} label="At Risk" value="1" color="#ef4444" />
            </div>

            <div className="grid-2">
                {/* Recovery Trajectory */}
                <div className="card">
                    <div className="card-header">
                        <div>
                            <div className="card-title">Recovery Trajectory</div>
                            <div className="card-subtitle">{selectedPatient} — Regional recovery scores</div>
                        </div>
                    </div>
                    <ResponsiveContainer width="100%" height={280}>
                        <AreaChart data={recoveryTimeline}>
                            <defs>
                                <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                                    <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                            <XAxis dataKey="timepoint" stroke="#64748b" fontSize={11} />
                            <YAxis stroke="#64748b" fontSize={11} domain={[0, 1]} />
                            <Tooltip contentStyle={{ background: '#1a1f2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
                            <Area type="monotone" dataKey="score" stroke="#10b981" fill="url(#scoreGrad)" strokeWidth={2.5} name="Overall" />
                            <Line type="monotone" dataKey="upper" stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Upper" />
                            <Line type="monotone" dataKey="middle" stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Middle" />
                            <Line type="monotone" dataKey="lower" stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Lower" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                {/* Regional Radar */}
                <div className="card">
                    <div className="card-header">
                        <div>
                            <div className="card-title">Regional Assessment</div>
                            <div className="card-subtitle">Latest timepoint metrics</div>
                        </div>
                    </div>
                    <ResponsiveContainer width="100%" height={280}>
                        <RadarChart data={radarData}>
                            <PolarGrid stroke="rgba(255,255,255,0.06)" />
                            <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                            <PolarRadiusAxis angle={30} domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
                            <Radar name="Recovery" dataKey="A" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.2} strokeWidth={2} />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Patient Timeline */}
            <div className="card">
                <div className="card-header">
                    <div className="card-title">Patient Recovery Overview</div>
                    <span className="badge badge-info">{patients.length} patients</span>
                </div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Patient</th>
                            <th>Trend</th>
                            <th>Recovery Score</th>
                            <th>Rate</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {patients.map(p => (
                            <tr key={p.id} onClick={() => setSelectedPatient(p.id)} style={{ cursor: 'pointer' }}>
                                <td style={{ fontWeight: 600, color: selectedPatient === p.id ? '#3b82f6' : '#f1f5f9' }}>{p.id}</td>
                                <td>
                                    <span style={{ color: p.trend === 'Improving' ? '#10b981' : p.trend === 'Declining' ? '#ef4444' : '#f59e0b' }}>
                                        {p.trend === 'Improving' ? '↑' : p.trend === 'Declining' ? '↓' : '→'} {p.trend}
                                    </span>
                                </td>
                                <td>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                        <div className="progress-bar" style={{ width: 100 }}>
                                            <div className="progress-fill" style={{
                                                width: `${p.score * 100}%`,
                                                background: p.score > 0.85 ? '#10b981' : p.score > 0.6 ? '#3b82f6' : p.score > 0.4 ? '#f59e0b' : '#ef4444'
                                            }} />
                                        </div>
                                        <span style={{ fontSize: 13, fontWeight: 600 }}>{(p.score * 100).toFixed(0)}%</span>
                                    </div>
                                </td>
                                <td style={{ color: p.rate.startsWith('-') ? '#ef4444' : '#10b981' }}>{p.rate}</td>
                                <td>
                                    <span className={`badge ${p.status === 'Complete' ? 'badge-success' : p.status === 'Partial' ? 'badge-purple' : p.status === 'Early' ? 'badge-info' : 'badge-warning'}`}>
                                        {p.status}
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}

function StatMini({ icon, label, value, color }) {
    return (
        <div className="stat-card">
            <div className="glow" style={{ background: `linear-gradient(90deg, ${color}, transparent)` }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div className="icon-wrap" style={{ background: `${color}15`, color, width: 36, height: 36 }}>{icon}</div>
                <div>
                    <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
                    <div style={{ fontSize: 22, fontWeight: 700 }}>{value}</div>
                </div>
            </div>
        </div>
    )
}
