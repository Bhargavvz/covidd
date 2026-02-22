import { BrowserRouter as Router, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  LayoutDashboard, Upload, Eye, Activity,
  BrainCircuit, Info, Stethoscope
} from 'lucide-react'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import UploadRegister from './pages/UploadRegister'
import ResultsViewer from './pages/ResultsViewer'
import RecoveryAnalysis from './pages/RecoveryAnalysis'
import ModelInfo from './pages/ModelInfo'
import About from './pages/About'
import './App.css'

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard', section: 'Overview' },
  { to: '/upload', icon: Upload, label: 'Upload & Register', section: 'Pipeline' },
  { to: '/results', icon: Eye, label: 'Results Viewer', section: 'Pipeline' },
  { to: '/recovery', icon: Activity, label: 'Recovery Analysis', section: 'Analysis' },
  { to: '/model', icon: BrainCircuit, label: 'Model Info', section: 'Analysis' },
  { to: '/about', icon: Info, label: 'About', section: 'System' },
]

function AppLayout() {
  const location = useLocation()
  const isLanding = location.pathname === '/'

  if (isLanding) {
    return <Landing />
  }

  const sections = [...new Set(navItems.map(n => n.section))]

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1><Stethoscope size={16} style={{ marginRight: 6, display: 'inline' }} />LungRecovery AI</h1>
          <span>Post-COVID CT Analysis</span>
        </div>
        <nav className="sidebar-nav">
          {sections.map(section => (
            <div key={section}>
              <div className="nav-section-label">{section}</div>
              {navItems.filter(n => n.section === section).map(item => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                >
                  <item.icon size={18} />
                  {item.label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="pipeline-status">
            <span className="status-dot" />
            Training in progress…
          </div>
        </div>
      </aside>
      <main className="main-content">
        <Routes>
          <Route path="/dashboard" element={<PageWrap><Dashboard /></PageWrap>} />
          <Route path="/upload" element={<PageWrap><UploadRegister /></PageWrap>} />
          <Route path="/results" element={<PageWrap><ResultsViewer /></PageWrap>} />
          <Route path="/recovery" element={<PageWrap><RecoveryAnalysis /></PageWrap>} />
          <Route path="/model" element={<PageWrap><ModelInfo /></PageWrap>} />
          <Route path="/about" element={<PageWrap><About /></PageWrap>} />
        </Routes>
      </main>
    </div>
  )
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/*" element={<AppLayout />} />
      </Routes>
    </Router>
  )
}

function PageWrap({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {children}
    </motion.div>
  )
}

export default App
