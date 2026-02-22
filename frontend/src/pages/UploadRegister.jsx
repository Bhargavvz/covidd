import { useState } from 'react'
import { motion } from 'framer-motion'
import { Upload, FileText, CheckCircle, AlertCircle, Loader, ArrowRight } from 'lucide-react'

const steps = [
    { id: 1, label: 'Upload CT Scans', desc: 'Upload moving and fixed NIfTI volumes' },
    { id: 2, label: 'Preprocessing', desc: 'Windowing, resampling, normalization' },
    { id: 3, label: 'Registration', desc: 'VoxelMorph deformable registration' },
    { id: 4, label: 'Analysis', desc: 'Jacobian, displacement, recovery metrics' },
]

export default function UploadRegister() {
    const [movingFile, setMovingFile] = useState(null)
    const [fixedFile, setFixedFile] = useState(null)
    const [running, setRunning] = useState(false)
    const [currentStep, setCurrentStep] = useState(0)
    const [complete, setComplete] = useState(false)

    const handleRun = () => {
        if (!movingFile || !fixedFile) return
        setRunning(true)
        setCurrentStep(1)
        // Simulate pipeline
        let step = 1
        const interval = setInterval(() => {
            step++
            if (step > 4) {
                clearInterval(interval)
                setRunning(false)
                setComplete(true)
                setCurrentStep(4)
            } else {
                setCurrentStep(step)
            }
        }, 2000)
    }

    return (
        <div className="page-container">
            <div className="page-header">
                <h2>Upload & Register</h2>
                <p>Upload CT volume pairs for deformable registration</p>
            </div>

            {/* Upload Zone */}
            <div className="grid-2" style={{ marginBottom: 24 }}>
                <UploadBox
                    label="Moving Volume (Baseline)"
                    sublabel="NIfTI (.nii.gz) format"
                    file={movingFile}
                    onFile={setMovingFile}
                />
                <UploadBox
                    label="Fixed Volume (Follow-up)"
                    sublabel="NIfTI (.nii.gz) format"
                    file={fixedFile}
                    onFile={setFixedFile}
                />
            </div>

            {/* Pipeline Steps */}
            <div className="card" style={{ marginBottom: 24 }}>
                <div className="card-header">
                    <div className="card-title">Registration Pipeline</div>
                    {complete && <span className="badge badge-success">✓ Complete</span>}
                </div>
                <div style={{ display: 'flex', gap: 0, alignItems: 'center' }}>
                    {steps.map((step, i) => (
                        <div key={step.id} style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
                            <motion.div
                                style={{
                                    flex: 1,
                                    padding: '16px',
                                    borderRadius: 10,
                                    background: currentStep >= step.id
                                        ? currentStep === step.id && running ? 'rgba(59,130,246,0.1)' : 'rgba(16,185,129,0.08)'
                                        : 'rgba(255,255,255,0.02)',
                                    border: `1px solid ${currentStep >= step.id ? (currentStep === step.id && running ? 'rgba(59,130,246,0.3)' : 'rgba(16,185,129,0.2)') : 'rgba(255,255,255,0.06)'}`,
                                    textAlign: 'center',
                                }}
                                animate={currentStep === step.id && running ? { scale: [1, 1.02, 1] } : {}}
                                transition={{ repeat: Infinity, duration: 1.5 }}
                            >
                                <div style={{ marginBottom: 6 }}>
                                    {currentStep > step.id || complete
                                        ? <CheckCircle size={20} color="#10b981" />
                                        : currentStep === step.id && running
                                            ? <Loader size={20} color="#3b82f6" style={{ animation: 'spin 1s linear infinite' }} />
                                            : <div style={{ width: 20, height: 20, borderRadius: '50%', border: '2px solid rgba(255,255,255,0.1)', margin: '0 auto' }} />
                                    }
                                </div>
                                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 2 }}>{step.label}</div>
                                <div style={{ fontSize: 11, color: '#64748b' }}>{step.desc}</div>
                            </motion.div>
                            {i < steps.length - 1 && (
                                <ArrowRight size={16} color="#64748b" style={{ margin: '0 4px', flexShrink: 0 }} />
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Run Button */}
            <div style={{ display: 'flex', gap: 12 }}>
                <button
                    className="btn btn-primary"
                    onClick={handleRun}
                    disabled={!movingFile || !fixedFile || running}
                    style={{ opacity: (!movingFile || !fixedFile || running) ? 0.5 : 1 }}
                >
                    {running ? <><Loader size={16} /> Processing...</> : 'Run Registration'}
                </button>
                {complete && (
                    <button className="btn btn-secondary" onClick={() => window.location.href = '/results'}>
                        View Results →
                    </button>
                )}
            </div>

            <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
        </div>
    )
}

function UploadBox({ label, sublabel, file, onFile }) {
    return (
        <div
            className="upload-zone"
            onClick={() => {
                const input = document.createElement('input')
                input.type = 'file'
                input.accept = '.nii,.nii.gz'
                input.onchange = (e) => onFile(e.target.files[0])
                input.click()
            }}
        >
            {file ? (
                <>
                    <div className="upload-icon" style={{ background: 'rgba(16,185,129,0.1)' }}>
                        <FileText size={24} color="#10b981" />
                    </div>
                    <h3 style={{ color: '#10b981' }}>{file.name}</h3>
                    <p>{(file.size / 1024 / 1024).toFixed(1)} MB — Click to replace</p>
                </>
            ) : (
                <>
                    <div className="upload-icon">
                        <Upload size={24} />
                    </div>
                    <h3>{label}</h3>
                    <p>{sublabel}</p>
                </>
            )}
        </div>
    )
}
