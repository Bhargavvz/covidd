import { motion } from 'framer-motion'
import {
    AreaChart, Area, BarChart, Bar, XAxis, YAxis,
    CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts'
import {
    Activity, Brain, Scan, TrendingDown, Clock,
    Database, Cpu, Zap
} from 'lucide-react'

const trainingData = [
    { epoch: 1, train: 0.82, val: 0.89 },
    { epoch: 10, train: 0.68, val: 0.74 },
    { epoch: 20, train: 0.52, val: 0.58 },
    { epoch: 30, train: 0.41, val: 0.47 },
    { epoch: 40, train: 0.33, val: 0.39 },
    { epoch: 50, train: 0.27, val: 0.33 },
    { epoch: 60, train: 0.22, val: 0.28 },
    { epoch: 70, train: 0.18, val: 0.24 },
    { epoch: 80, train: 0.15, val: 0.21 },
    { epoch: 90, train: 0.13, val: 0.18 },
    { epoch: 100, train: 0.11, val: 0.16 },
]

const pipelineStats = [
    { name: 'Completed', value: 78, color: '#10b981' },
    { name: 'In Progress', value: 15, color: '#3b82f6' },
    { name: 'Queued', value: 7, color: '#64748b' },
]

const recentScans = [
    { id: 'PAT-001', date: '2024-01-15', ncc: 0.92, status: 'Complete Recovery' },
    { id: 'PAT-002', date: '2024-01-14', ncc: 0.78, status: 'Partial Recovery' },
    { id: 'PAT-003', date: '2024-01-14', ncc: 0.65, status: 'Early Recovery' },
    { id: 'PAT-004', date: '2024-01-13', ncc: 0.88, status: 'Complete Recovery' },
    { id: 'PAT-005', date: '2024-01-12', ncc: 0.71, status: 'Partial Recovery' },
]

export default function Dashboard() {
    return (
        <div className="page-container">
            {/* Hero */}
            <div className="hero-section">
                <div className="hero-content">
                    <div className="hero-badge">
                        <Zap size={12} /> AI-Powered Medical Imaging
                    </div>
                    <h1 className="hero-title">
                        Longitudinal Analysis of <br />
                        <span className="gradient-text">Post-COVID Lung Recovery</span>
                    </h1>
                    <p className="hero-description">
                        Deformable CT image registration with diffeomorphic VoxelMorph for
                        tracking and analyzing post-COVID-19 lung tissue recovery over time.
                    </p>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="stats-grid">
                <StatCard
                    icon={<Database size={20} />}
                    value="576"
                    label="Training Pairs"
                    change="+192 synthetic"
                    positive
                    color="#3b82f6"
                />
                <StatCard
                    icon={<Brain size={20} />}
                    value="5.8M"
                    label="Model Parameters"
                    change="VoxelMorphDiff"
                    positive
                    color="#8b5cf6"
                />
                <StatCard
                    icon={<TrendingDown size={20} />}
                    value="0.16"
                    label="Best Val Loss (NCC)"
                    change="−82% from start"
                    positive
                    color="#10b981"
                />
                <StatCard
                    icon={<Clock size={20} />}
                    value="3.4h"
                    label="Training Time"
                    change="H200 GPU"
                    positive
                    color="#06b6d4"
                />
            </div>

            <div className="grid-2">
                {/* Training Curve */}
                <div className="card">
                    <div className="card-header">
                        <div>
                            <div className="card-title">Training Progress</div>
                            <div className="card-subtitle">NCC Loss over 100 epochs</div>
                        </div>
                        <span className="badge badge-success">100/100</span>
                    </div>
                    <ResponsiveContainer width="100%" height={240}>
                        <AreaChart data={trainingData}>
                            <defs>
                                <linearGradient id="trainGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                                    <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="valGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                            <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                            <YAxis stroke="#64748b" fontSize={11} />
                            <Tooltip
                                contentStyle={{ background: '#1a1f2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }}
                                labelStyle={{ color: '#94a3b8' }}
                            />
                            <Area type="monotone" dataKey="train" stroke="#3b82f6" fill="url(#trainGrad)" strokeWidth={2} name="Train" />
                            <Area type="monotone" dataKey="val" stroke="#8b5cf6" fill="url(#valGrad)" strokeWidth={2} name="Val" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                {/* Pipeline Status */}
                <div className="card">
                    <div className="card-header">
                        <div>
                            <div className="card-title">Pipeline Status</div>
                            <div className="card-subtitle">Registration workload</div>
                        </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 30 }}>
                        <ResponsiveContainer width={160} height={160}>
                            <PieChart>
                                <Pie
                                    data={pipelineStats}
                                    innerRadius={50}
                                    outerRadius={70}
                                    paddingAngle={4}
                                    dataKey="value"
                                >
                                    {pipelineStats.map((entry, i) => (
                                        <Cell key={i} fill={entry.color} />
                                    ))}
                                </Pie>
                            </PieChart>
                        </ResponsiveContainer>
                        <div>
                            {pipelineStats.map(s => (
                                <div key={s.name} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: s.color }} />
                                    <span style={{ fontSize: 13, color: '#94a3b8' }}>{s.name}</span>
                                    <span style={{ fontWeight: 600, fontSize: 14, marginLeft: 'auto' }}>{s.value}%</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Recent Scans */}
            <div className="card">
                <div className="card-header">
                    <div>
                        <div className="card-title">Recent Registrations</div>
                        <div className="card-subtitle">Latest patient registration results</div>
                    </div>
                    <span className="badge badge-info">5 results</span>
                </div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Patient ID</th>
                            <th>Date</th>
                            <th>NCC Score</th>
                            <th>Recovery Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {recentScans.map(scan => (
                            <tr key={scan.id}>
                                <td style={{ fontWeight: 600, color: '#f1f5f9' }}>{scan.id}</td>
                                <td>{scan.date}</td>
                                <td>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                        <div className="progress-bar" style={{ width: 80 }}>
                                            <div
                                                className="progress-fill"
                                                style={{ width: `${scan.ncc * 100}%`, background: scan.ncc > 0.85 ? '#10b981' : scan.ncc > 0.7 ? '#f59e0b' : '#ef4444' }}
                                            />
                                        </div>
                                        <span style={{ fontSize: 13, fontWeight: 600 }}>{scan.ncc.toFixed(2)}</span>
                                    </div>
                                </td>
                                <td>
                                    <span className={`badge ${scan.status.includes('Complete') ? 'badge-success' : scan.status.includes('Partial') ? 'badge-warning' : 'badge-info'}`}>
                                        {scan.status}
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

function StatCard({ icon, value, label, change, positive, color }) {
    return (
        <motion.div
            className="stat-card"
            whileHover={{ y: -2 }}
            transition={{ duration: 0.2 }}
        >
            <div className="glow" style={{ background: `linear-gradient(90deg, ${color}, transparent)` }} />
            <div className="icon-wrap" style={{ background: `${color}15`, color }}>
                {icon}
            </div>
            <div className="stat-value">{value}</div>
            <div className="stat-label">{label}</div>
            {change && (
                <div className={`stat-change ${positive ? 'positive' : 'negative'}`}>
                    {change}
                </div>
            )}
        </motion.div>
    )
}
