import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, ReferenceLine 
} from 'recharts';
import { 
  ShieldAlert, Activity, Satellite, RadioTower, AlertTriangle, Zap, Server, Clock 
} from 'lucide-react';

// Simulated Data (T-6 hours to T+12 hours prediction)
const generateData = () => {
  const data = [];
  const now = new Date();
  
  // Historical data (past 6 hours)
  for(let i = -36; i <= 0; i++) {
    const time = new Date(now.getTime() + i * 10 * 60000);
    const baseVal = Math.pow(Math.random() * 2, 2); 
    data.push({
      time: time.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
      actual: baseVal + Math.random() * 0.5,
      predicted: null,
      upperBound: null,
      lowerBound: null,
      isPrediction: false
    });
  }
  
  // Last actual value for continuity
  const lastActual = data[data.length - 1].actual;
  data[data.length - 1].predicted = lastActual;
  
  // Prediction data (next 12 hours) with a massive Solar Storm spike at T+45 mins
  for(let i = 1; i <= 72; i++) {
    const time = new Date(now.getTime() + i * 10 * 60000);
    let predVal = lastActual + Math.random();
    
    // Solar storm spike around i = 4 to 12 (T+40 mins to T+120 mins)
    if (i >= 4 && i <= 15) {
      predVal += Math.pow(i - 3, 2) * 5 * Math.random(); 
    } else if (i > 15) {
      predVal += 15 * Math.exp(-(i-15)/10);
    }
    
    data.push({
      time: time.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
      actual: null,
      predicted: predVal,
      upperBound: predVal * 1.3,
      lowerBound: predVal * 0.7,
      isPrediction: true
    });
  }
  return data;
};

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: 'rgba(16, 21, 34, 0.9)',
        border: '1px solid var(--border-light)',
        padding: '12px',
        borderRadius: '8px',
        backdropFilter: 'blur(10px)'
      }}>
        <p style={{ color: 'var(--text-muted)', marginBottom: '8px' }}>{label}</p>
        {payload.map((entry, index) => (
          <div key={index} style={{ color: entry.color, fontWeight: 'bold' }}>
            {entry.name}: {entry.value.toFixed(2)} MeV
          </div>
        ))}
      </div>
    );
  }
  return null;
};

function App() {
  const [data, setData] = useState([]);
  const [criticalAlert, setCriticalAlert] = useState(false);

  useEffect(() => {
    const freshData = generateData();
    setData(freshData);
    
    // Check if any prediction crosses danger threshold
    const hasDanger = freshData.some(d => d.predicted > 50);
    setCriticalAlert(hasDanger);
  }, []);

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="brand-container">
          <ShieldAlert size={32} className="brand-icon" />
          <div className="brand">PRAHARI</div>
        </div>
        
        <div className="nav-item active">
          <Activity size={20} />
          <span>Real-time Monitoring</span>
        </div>
        <div className="nav-item">
          <Satellite size={20} />
          <span>Orbital Assets</span>
        </div>
        <div className="nav-item">
          <RadioTower size={20} />
          <span>Ground Infrastructure</span>
        </div>
        <div className="nav-item">
          <Server size={20} />
          <span>UPI / Banking Systems</span>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="header">
          <div>
            <h2>ISRO Early Warning System</h2>
            <p style={{color: 'var(--text-muted)', marginTop: '4px'}}>
              Geostationary Orbit Electron Flux Forecast
            </p>
          </div>
          
          <div className={`status-badge ${criticalAlert ? 'critical' : ''}`}>
            {criticalAlert ? <AlertTriangle size={20} /> : <Zap size={20} />}
            {criticalAlert ? 'CRITICAL STORM ALERT' : 'SYSTEM NOMINAL'}
          </div>
        </div>

        {/* Alert Banner */}
        {criticalAlert && (
          <div className="alert-panel">
            <AlertTriangle size={48} className="alert-icon" />
            <div className="alert-content">
              <h3>SEVERE SOLAR RADIATION DETECTED IN FORECAST</h3>
              <p>Moirai AI predicts Class-X radiation flux crossing 100 MeV at T+45 mins. 
                <strong> High risk of disruption to GSAT-10 (Banking/UPI) and NavIC.</strong>
              </p>
            </div>
            <div style={{marginLeft: 'auto'}}>
              <button style={{
                background: 'var(--accent-warning)',
                color: '#fff',
                border: 'none',
                padding: '12px 24px',
                borderRadius: '6px',
                fontWeight: 'bold',
                cursor: 'pointer',
                letterSpacing: '1px'
              }}>
                INITIATE SAFE MODE
              </button>
            </div>
          </div>
        )}

        {/* Stats Grid */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-title">Current Flux Level</div>
            <div className="stat-value">2.4 <span style={{fontSize: '1rem', color: 'var(--text-muted)'}}>MeV</span></div>
          </div>
          <div className="stat-card">
            <div className="stat-title">Peak Forecast (12h)</div>
            <div className="stat-value warning">118.5 <span style={{fontSize: '1rem', color: 'var(--text-muted)'}}>MeV</span></div>
          </div>
          <div className="stat-card">
            <div className="stat-title">Time to Impact</div>
            <div className="stat-value warning" style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
              <Clock size={28} /> 00:45:00
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-title">Satellites at Risk</div>
            <div className="stat-value">3</div>
          </div>
        </div>

        {/* Chart Section */}
        <div className="chart-section">
          <div className="chart-header">
            <h3>Radiation Flux Forecast (Moirai Model)</h3>
            <div style={{display: 'flex', gap: '16px'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)'}}>
                <div style={{width: '12px', height: '12px', background: 'var(--chart-line)', borderRadius: '50%'}}></div>
                Actual Readings
              </div>
              <div style={{display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)'}}>
                <div style={{width: '12px', height: '12px', background: 'var(--chart-pred)', borderRadius: '50%'}}></div>
                AI Prediction
              </div>
            </div>
          </div>
          
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis 
                dataKey="time" 
                stroke="var(--text-muted)" 
                minTickGap={50}
                tick={{fill: 'var(--text-muted)'}}
              />
              <YAxis 
                stroke="var(--text-muted)"
                tick={{fill: 'var(--text-muted)'}}
                label={{ value: 'Electron Flux (>2 MeV)', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)' }}
              />
              <Tooltip content={<CustomTooltip />} />
              
              <ReferenceLine y={50} stroke="var(--accent-warning)" strokeDasharray="3 3" label={{ position: 'top', value: 'Danger Threshold', fill: 'var(--accent-warning)' }} />
              
              <Line 
                type="monotone" 
                dataKey="actual" 
                stroke="var(--chart-line)" 
                strokeWidth={3} 
                dot={false}
                name="Actual Flux"
              />
              <Line 
                type="monotone" 
                dataKey="predicted" 
                stroke="var(--chart-pred)" 
                strokeWidth={3} 
                dot={false}
                strokeDasharray="5 5"
                name="Predicted Flux"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default App;
