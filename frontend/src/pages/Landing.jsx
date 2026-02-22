import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
    Stethoscope, ArrowRight, Brain, Activity, Scan,
    BarChart3, Shield, Zap, ChevronDown, Github
} from 'lucide-react'

const features = [
    { icon: <Scan size={24} />, title: 'CT Registration', desc: 'Diffeomorphic VoxelMorph aligns lung scans across timepoints with anatomically plausible deformations', color: '#3b82f6' },
    { icon: <Brain size={24} />, title: 'Deep Learning', desc: '5.8M parameter 3D U-Net trained on H200 GPU with BFloat16 mixed precision for fast inference', color: '#8b5cf6' },
    { icon: <Activity size={24} />, title: 'Recovery Tracking', desc: 'Quantitative Jacobian analysis measures tissue expansion, contraction, and regional recovery over time', color: '#06b6d4' },
    { icon: <BarChart3 size={24} />, title: 'Visual Analytics', desc: 'Interactive dashboards with training curves, deformation grids, heatmaps, and recovery trajectories', color: '#10b981' },
]

const stats = [
    { value: '576', label: 'Training Pairs' },
    { value: '5.8M', label: 'Parameters' },
    { value: '128³', label: 'Volume Size' },
    { value: 'H200', label: 'GPU Trained' },
]

const pipeline = [
    { step: '01', title: 'Acquire', desc: 'Chest CT scans at multiple timepoints' },
    { step: '02', title: 'Preprocess', desc: 'Window, resample, normalize to 128³' },
    { step: '03', title: 'Register', desc: 'VoxelMorph predicts displacement field' },
    { step: '04', title: 'Analyze', desc: 'Recovery scores & regional assessment' },
]

export default function Landing() {
    const navigate = useNavigate()

    return (
        <div style={{ minHeight: '100vh', background: '#0a0e1a', overflow: 'auto' }}>
            {/* Nav */}
            <nav style={{
                position: 'fixed', top: 0, left: 0, right: 0, zIndex: 50,
                padding: '16px 40px',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                background: 'rgba(10, 14, 26, 0.8)', backdropFilter: 'blur(12px)',
                borderBottom: '1px solid rgba(255,255,255,0.04)',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Stethoscope size={20} color="#3b82f6" />
                    <span style={{ fontSize: 16, fontWeight: 700, background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                        LungRecovery AI
                    </span>
                </div>
                <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
                    <a href="#features" style={{ color: '#94a3b8', textDecoration: 'none', fontSize: 13, fontWeight: 500 }}>Features</a>
                    <a href="#pipeline" style={{ color: '#94a3b8', textDecoration: 'none', fontSize: 13, fontWeight: 500 }}>Pipeline</a>
                    <a href="#tech" style={{ color: '#94a3b8', textDecoration: 'none', fontSize: 13, fontWeight: 500 }}>Tech</a>
                    <button
                        className="btn btn-primary"
                        onClick={() => navigate('/dashboard')}
                        style={{ padding: '8px 18px', fontSize: 13 }}
                    >
                        Open Dashboard <ArrowRight size={14} />
                    </button>
                </div>
            </nav>

            {/* Hero */}
            <section style={{
                minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
                position: 'relative', overflow: 'hidden', textAlign: 'center', padding: '0 40px',
            }}>
                {/* Animated background orbs */}
                <div style={{ position: 'absolute', inset: 0, overflow: 'hidden' }}>
                    <motion.div
                        animate={{ x: [0, 30, 0], y: [0, -20, 0] }}
                        transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
                        style={{
                            position: 'absolute', top: '10%', right: '15%',
                            width: 500, height: 500, borderRadius: '50%',
                            background: 'radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%)',
                        }}
                    />
                    <motion.div
                        animate={{ x: [0, -20, 0], y: [0, 30, 0] }}
                        transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
                        style={{
                            position: 'absolute', bottom: '15%', left: '10%',
                            width: 400, height: 400, borderRadius: '50%',
                            background: 'radial-gradient(circle, rgba(139,92,246,0.06) 0%, transparent 70%)',
                        }}
                    />
                    <motion.div
                        animate={{ x: [0, 15, 0], y: [0, 15, 0] }}
                        transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
                        style={{
                            position: 'absolute', top: '40%', left: '30%',
                            width: 300, height: 300, borderRadius: '50%',
                            background: 'radial-gradient(circle, rgba(6,182,212,0.05) 0%, transparent 70%)',
                        }}
                    />
                </div>

                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                    style={{ position: 'relative', zIndex: 1, maxWidth: 800 }}
                >
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.2 }}
                        style={{
                            display: 'inline-flex', alignItems: 'center', gap: 8,
                            padding: '8px 18px', borderRadius: 24,
                            background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)',
                            fontSize: 13, fontWeight: 500, color: '#3b82f6', marginBottom: 24,
                        }}
                    >
                        <Zap size={14} /> AI-Powered Medical Imaging Research
                    </motion.div>

                    <h1 style={{
                        fontSize: 56, fontWeight: 800, letterSpacing: '-0.04em',
                        lineHeight: 1.1, marginBottom: 20, color: '#f1f5f9',
                    }}>
                        Tracking Lung Recovery<br />
                        <span style={{
                            background: 'linear-gradient(135deg, #06b6d4, #3b82f6, #8b5cf6)',
                            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                        }}>After COVID-19</span>
                    </h1>

                    <p style={{
                        fontSize: 17, color: '#94a3b8', lineHeight: 1.7,
                        maxWidth: 580, margin: '0 auto 36px',
                    }}>
                        Deformable CT image registration with diffeomorphic VoxelMorph for
                        longitudinal analysis of post-COVID-19 lung tissue recovery patterns.
                    </p>

                    <div style={{ display: 'flex', gap: 14, justifyContent: 'center' }}>
                        <motion.button
                            className="btn btn-primary"
                            whileHover={{ scale: 1.03 }}
                            whileTap={{ scale: 0.97 }}
                            onClick={() => navigate('/dashboard')}
                            style={{ padding: '14px 32px', fontSize: 15, borderRadius: 10 }}
                        >
                            Launch Dashboard <ArrowRight size={16} />
                        </motion.button>
                        <motion.a
                            href="https://github.com/Bhargavvz/covidd"
                            target="_blank"
                            rel="noreferrer"
                            className="btn btn-secondary"
                            whileHover={{ scale: 1.03 }}
                            style={{ padding: '14px 28px', fontSize: 15, borderRadius: 10, textDecoration: 'none' }}
                        >
                            <Github size={16} /> View Source
                        </motion.a>
                    </div>

                    {/* Scroll indicator */}
                    <motion.div
                        animate={{ y: [0, 8, 0] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        style={{ marginTop: 60, color: '#475569' }}
                    >
                        <ChevronDown size={20} />
                    </motion.div>
                </motion.div>
            </section>

            {/* Stats Bar */}
            <section style={{
                padding: '40px 0',
                background: 'rgba(255,255,255,0.02)',
                borderTop: '1px solid rgba(255,255,255,0.04)',
                borderBottom: '1px solid rgba(255,255,255,0.04)',
            }}>
                <div style={{ maxWidth: 900, margin: '0 auto', display: 'flex', justifyContent: 'space-around' }}>
                    {stats.map((s, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.1 }}
                            viewport={{ once: true }}
                            style={{ textAlign: 'center' }}
                        >
                            <div style={{ fontSize: 36, fontWeight: 800, letterSpacing: '-0.03em', background: 'linear-gradient(135deg, #3b82f6, #06b6d4)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>{s.value}</div>
                            <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>{s.label}</div>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* Features */}
            <section id="features" style={{ padding: '80px 40px', maxWidth: 1100, margin: '0 auto' }}>
                <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    style={{ textAlign: 'center', marginBottom: 48 }}
                >
                    <h2 style={{ fontSize: 32, fontWeight: 700, letterSpacing: '-0.02em', marginBottom: 10 }}>Core Capabilities</h2>
                    <p style={{ fontSize: 15, color: '#64748b' }}>End-to-end pipeline from CT acquisition to recovery analysis</p>
                </motion.div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 20 }}>
                    {features.map((f, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.1 }}
                            viewport={{ once: true }}
                            className="card"
                            style={{ padding: 28 }}
                        >
                            <div style={{
                                width: 48, height: 48, borderRadius: 12,
                                background: `${f.color}12`, color: f.color,
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                marginBottom: 16,
                            }}>{f.icon}</div>
                            <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>{f.title}</h3>
                            <p style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.6 }}>{f.desc}</p>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* Pipeline */}
            <section id="pipeline" style={{ padding: '80px 40px', background: 'rgba(255,255,255,0.01)' }}>
                <div style={{ maxWidth: 900, margin: '0 auto' }}>
                    <motion.div
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                        style={{ textAlign: 'center', marginBottom: 48 }}
                    >
                        <h2 style={{ fontSize: 32, fontWeight: 700, letterSpacing: '-0.02em', marginBottom: 10 }}>How It Works</h2>
                        <p style={{ fontSize: 15, color: '#64748b' }}>Four-stage registration and analysis pipeline</p>
                    </motion.div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
                        {pipeline.map((p, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.15 }}
                                viewport={{ once: true }}
                                style={{
                                    padding: 24, borderRadius: 12, textAlign: 'center',
                                    background: 'var(--bg-card)', border: '1px solid var(--border)',
                                    position: 'relative',
                                }}
                            >
                                <div style={{
                                    fontSize: 32, fontWeight: 800, color: 'rgba(59,130,246,0.15)',
                                    position: 'absolute', top: 10, right: 14,
                                }}>{p.step}</div>
                                <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 6, marginTop: 8 }}>{p.title}</div>
                                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{p.desc}</div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Tech Stack */}
            <section id="tech" style={{ padding: '80px 40px', maxWidth: 800, margin: '0 auto', textAlign: 'center' }}>
                <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
                    <h2 style={{ fontSize: 32, fontWeight: 700, marginBottom: 24 }}>Built With</h2>
                    <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 10 }}>
                        {['PyTorch 2.x', 'VoxelMorph', 'SimpleITK', 'NumPy', 'SciPy', 'React', 'Vite', 'Recharts', 'Framer Motion', 'NVIDIA H200', 'BFloat16 AMP', 'STOIC-2021'].map(t => (
                            <span key={t} style={{
                                padding: '8px 16px', borderRadius: 8, fontSize: 13, fontWeight: 500,
                                background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.15)',
                                color: '#94a3b8',
                            }}>{t}</span>
                        ))}
                    </div>
                </motion.div>
            </section>

            {/* CTA */}
            <section style={{
                padding: '80px 40px', textAlign: 'center',
                background: 'linear-gradient(180deg, transparent 0%, rgba(59,130,246,0.04) 100%)',
            }}>
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                >
                    <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 12 }}>Ready to Explore?</h2>
                    <p style={{ fontSize: 15, color: '#64748b', marginBottom: 24 }}>
                        View training results, registration outputs, and recovery analysis
                    </p>
                    <motion.button
                        className="btn btn-primary"
                        whileHover={{ scale: 1.03 }}
                        whileTap={{ scale: 0.97 }}
                        onClick={() => navigate('/dashboard')}
                        style={{ padding: '14px 36px', fontSize: 15, borderRadius: 10 }}
                    >
                        Launch Dashboard <ArrowRight size={16} />
                    </motion.button>
                </motion.div>
            </section>

            {/* Footer */}
            <footer style={{
                padding: '24px 40px', textAlign: 'center', fontSize: 12, color: '#475569',
                borderTop: '1px solid rgba(255,255,255,0.04)',
            }}>
                COVID-19 Lung Recovery Analysis © 2024 — Built with ❤️ for medical AI research
            </footer>
        </div>
    )
}
