import { Heart, Github, BookOpen, Shield, Scan, Brain, BarChart3, Activity } from 'lucide-react'

const methodology = [
    { icon: <Scan size={16} />, title: 'CT Acquisition', desc: 'Collect chest CT scans at multiple timepoints from COVID-19 recovered patients', color: '#3b82f6' },
    { icon: <Brain size={16} />, title: 'Preprocessing', desc: 'Lung windowing (W:1500, L:-600), isotropic resampling to 1.5mm, intensity normalization to [0,1]', color: '#8b5cf6' },
    { icon: <Activity size={16} />, title: 'Deformable Registration', desc: 'VoxelMorph predicts a velocity field, integrated via scaling & squaring for diffeomorphic displacement', color: '#06b6d4' },
    { icon: <BarChart3 size={16} />, title: 'Recovery Analysis', desc: 'Jacobian determinants quantify tissue expansion/contraction; regional scores track upper/middle/lower lung recovery', color: '#10b981' },
]

const techStack = [
    'PyTorch 2.x', 'SimpleITK', 'SciPy', 'NumPy',
    'React + Vite', 'Recharts', 'Framer Motion',
    'NVIDIA H200 GPU', 'BFloat16 AMP', 'STOIC-2021 Dataset',
]

const references = [
    { title: 'VoxelMorph: Learning-based Image Registration', venue: 'IEEE TMI 2019', authors: 'Balakrishnan et al.' },
    { title: 'Diffeomorphic Registration with Scaling and Squaring', venue: 'NeuroImage 2005', authors: 'Arsigny et al.' },
    { title: 'STOIC-2021: COVID-19 AI Challenge', venue: 'Radiology 2021', authors: 'Defined et al.' },
    { title: 'Post-COVID Lung Recovery Patterns', venue: 'Lancet Resp Med 2022', authors: 'Wu et al.' },
]

export default function About() {
    return (
        <div className="page-container">
            <div className="page-header">
                <h2>About This Project</h2>
                <p>AI-Based Longitudinal Analysis of Post-COVID-19 Lung Recovery Using Deformable CT Image Registration</p>
            </div>

            {/* Project Description */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-title" style={{ marginBottom: 12 }}>
                    <Heart size={16} style={{ marginRight: 6, color: '#ef4444' }} />
                    Project Overview
                </div>
                <p style={{ fontSize: 14, color: '#94a3b8', lineHeight: 1.7 }}>
                    This project develops a deep learning pipeline for tracking lung tissue recovery in
                    post-COVID-19 patients using deformable CT image registration. By analyzing structural
                    changes in lung tissue over time through diffeomorphic registration, we can quantitatively
                    measure how patients' lungs recover from COVID-19 damage — identifying regions of
                    improvement, stagnation, or deterioration.
                </p>
                <p style={{ fontSize: 14, color: '#94a3b8', lineHeight: 1.7, marginTop: 12 }}>
                    The pipeline uses a VoxelMorph-based architecture with diffeomorphic constraints to ensure
                    anatomically plausible deformations. Jacobian determinant analysis reveals tissue expansion
                    (recovery) and contraction patterns, while regional scoring enables localized assessment
                    of upper, middle, and lower lung zones.
                </p>
            </div>

            {/* Methodology */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-title" style={{ marginBottom: 18 }}>Methodology</div>
                <div className="method-steps">
                    {methodology.map((step, i) => (
                        <div className="method-step" key={i}>
                            <div className="step-number" style={{ background: `${step.color}15`, color: step.color }}>
                                {step.icon}
                            </div>
                            <div className="step-content">
                                <h4>{step.title}</h4>
                                <p>{step.desc}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="grid-2">
                {/* Tech Stack */}
                <div className="card">
                    <div className="card-title" style={{ marginBottom: 14 }}>
                        <Shield size={15} style={{ marginRight: 6 }} />
                        Technology Stack
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                        {techStack.map(tech => (
                            <span key={tech} className="badge badge-info" style={{ padding: '6px 12px', fontSize: 12 }}>
                                {tech}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Key References */}
                <div className="card">
                    <div className="card-title" style={{ marginBottom: 14 }}>
                        <BookOpen size={15} style={{ marginRight: 6 }} />
                        Key References
                    </div>
                    {references.map((ref, i) => (
                        <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < references.length - 1 ? '1px solid rgba(255,255,255,0.06)' : 'none' }}>
                            <div style={{ fontSize: 13, fontWeight: 600, color: '#f1f5f9' }}>{ref.title}</div>
                            <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{ref.authors} — <em>{ref.venue}</em></div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Footer */}
            <div style={{
                textAlign: 'center', padding: '32px 0 16px', color: '#475569', fontSize: 12
            }}>
                <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginBottom: 8 }}>
                    <a href="https://github.com/Bhargavvz/covidd" target="_blank" rel="noreferrer" style={{ color: '#94a3b8', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 4 }}>
                        <Github size={14} /> GitHub Repository
                    </a>
                </div>
                COVID-19 Lung Recovery Analysis © 2024
            </div>
        </div>
    )
}
