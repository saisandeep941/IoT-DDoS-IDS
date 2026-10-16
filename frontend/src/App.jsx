import axios from "axios";
import { useRef } from "react";
import { useState, useEffect } from "react";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

import {
  ShieldCheck, LayoutDashboard, Activity, ShieldAlert,
  Network, Settings, Search, Bell, BrainCircuit,
  CircleCheck, Menu, ArrowUpRight
} from "lucide-react";
import "./App.css";

const navItems = [
  { name: "Overview", icon: LayoutDashboard },
  { name: "Network Traffic", icon: Activity },
  { name: "Threat Detection", icon: ShieldAlert },
  { name: "AI Insights", icon: BrainCircuit },
];

function App() {

  const [active, setActive] = useState("Overview");
  const [search, setSearch] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [apiStatus, setApiStatus] = useState("Checking...");
  const [prediction, setPrediction] = useState(null);
  const [streamResult, setStreamResult] = useState(null);
const [streamError, setStreamError] = useState("");
const [streamLoading, setStreamLoading] = useState(false);
const [streamActive, setStreamActive] = useState(false);
const [streamCount, setStreamCount] = useState(0);
  const [riskResult, setRiskResult] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [detectionHistory, setDetectionHistory] = useState([]);
  const [projectInfo, setProjectInfo] = useState(null);
  const [modelPrediction, setModelPrediction] = useState(null);
  const [shapData, setShapData] = useState(null);
const [shapError, setShapError] = useState("");

const [csvStreamLoading, setCsvStreamLoading] = useState(false);
const [csvStreamError, setCsvStreamError] = useState("");
const [csvStreamResult, setCsvStreamResult] = useState(null);

const runCsvStream = async (file) => {
  if (!file) return;

  setCsvStreamLoading(true);
  setCsvStreamError("");
  setCsvStreamResult(null);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post(
      "http://127.0.0.1:8000/api/stream/csv",
      formData
    );

    setCsvStreamResult(response.data);
  } catch (error) {
    const detail = error.response?.data?.detail;

    setCsvStreamError(
      typeof detail === "string"
        ? detail
        : detail?.message || error.message || "CSV stream processing failed."
    );
  } finally {
    setCsvStreamLoading(false);
  }
};



  const [dashboardData, setDashboardData] = useState(null);
  const [trafficAnalysis, setTrafficAnalysis] = useState(null);

  
const stats = [
  {
    title: "Total Traffic",
    value: trafficAnalysis?.total_traffic?.toLocaleString() ?? "Upload dataset",
    icon: Network,
  },
  {
    title: "Attacks Detected",
    
value: trafficAnalysis?.attacks_detected?.toLocaleString() ?? "Upload dataset",

    icon: ShieldAlert,
  },
  {
  title: "Blocking Status",
  value: "Not Implemented",
  icon: ShieldCheck,
},
  

{
  title: "Attack Traffic",
  value: trafficAnalysis
    ? `${trafficAnalysis.total_traffic > 0
        ? (
            (trafficAnalysis.attacks_detected /
              trafficAnalysis.total_traffic) * 100
          ).toFixed(1)
        : "0.0"}%`
    : "Upload dataset",
  icon: Activity,
},


];

const fileInputRef = useRef(null);
const [uploadResult, setUploadResult] = useState(null);
const [uploadError, setUploadError] = useState("");
const [uploading, setUploading] = useState(false);
const [modelStatus, setModelStatus] = useState(null);

useEffect(() => {
  axios
    .get("http://127.0.0.1:8000/api/model/status")
    .then((response) => setModelStatus(response.data))
    .catch((error) => {
      console.error("Model status API error:", error);
      setModelStatus(null);
    });
}, []);





useEffect(() => {
  const fetchData = async () => {
    try {
      const [historyResponse, projectResponse, dashboardResponse] =
        await Promise.all([
          axios.get("http://127.0.0.1:8000/api/detection-history"),
          axios.get("http://127.0.0.1:8000/api/project/info"),
          axios.get("http://127.0.0.1:8000/api/dashboard"),
        ]);

      setDetectionHistory(historyResponse.data.history);
      setProjectInfo(projectResponse.data);
      setDashboardData(dashboardResponse.data);
    } catch (error) {
      console.error("Dashboard refresh error:", error);
    }
  };

  fetchData();

  const intervalId = setInterval(fetchData, 5000);

  return () => clearInterval(intervalId);
}, []);


const uploadDataset = async (file) => {
  if (!file) return;

  setUploading(true);
  setUploadError("");
  setUploadResult(null);

  const formData = new FormData();
  formData.append("file", file);

  
try {
  const response = await axios.post(
    "http://127.0.0.1:8000/api/dataset/preview",
    formData
  );
  setUploadResult(response.data);

  const analysisResponse = await axios.post(
    "http://127.0.0.1:8000/api/dataset/analyze",
    formData
  );
  setTrafficAnalysis(analysisResponse.data);
  
const predictionForm = new FormData();
predictionForm.append("file", file);

const predictionResponse = await axios.post(
  "http://127.0.0.1:8000/api/predict/csv",
  predictionForm
);

setModelPrediction(predictionResponse.data);

  
setDashboardData((prev) => ({
  ...prev,
  total_traffic: analysisResponse.data.total_traffic,
  attacks_detected: analysisResponse.data.attacks_detected,
  threats_blocked: 0,
  detection_rate:
    analysisResponse.data.total_traffic > 0
      ? Number(
          (
            (analysisResponse.data.attacks_detected /
              analysisResponse.data.total_traffic) *
            100
          ).toFixed(1)
        )
      : 0,
}));


} catch (error) {
  setUploadError(
    error.response?.data?.detail || "Upload failed."
  );
} finally {
  setUploading(false);
}

};

const [backendAlerts, setBackendAlerts] = useState([]);

useEffect(() => {
  axios
    .get("http://127.0.0.1:8000/api/alerts")
    .then((response) => setBackendAlerts(response.data.alerts))
    .catch((error) => console.error("Alerts API error:", error));
}, []);


useEffect(() => {
  axios.get("http://127.0.0.1:8000/api/health")
    .then((response) => setApiStatus(response.data.status))
    .catch(() => setApiStatus("Disconnected"));
}, []);

  
const filteredAlerts = detectionHistory.filter((alert) =>
  `${alert.prediction} ${alert.risk_score} ${alert.severity} ${alert.source} ${alert.timestamp}`
    .toLowerCase()
    .includes(search.toLowerCase())
);

  const [systemSummary, setSystemSummary] = useState(null);

useEffect(() => {
  axios
    .get("http://127.0.0.1:8000/api/system/summary")
    .then((response) => setSystemSummary(response.data))
    .catch(() => setSystemSummary(null));
}, []);
const [systemMetrics, setSystemMetrics] = useState(null);

useEffect(() => {
  axios.get("http://127.0.0.1:8000/api/system/metrics")
    .then((res) => setSystemMetrics(res.data))
    .catch(() => setSystemMetrics(null));
}, []);

const [serverTime, setServerTime] = useState(null);

useEffect(() => {
  axios.get("http://127.0.0.1:8000/api/system/time")
    .then((res) => setServerTime(res.data))
    .catch(() => setServerTime(null));
}, []);

useEffect(() => {
  if (active !== "AI Insights") return;

  const fetchShapData = async () => {
    try {
      setShapError("");
      const response = await axios.get(
        "http://127.0.0.1:8000/api/model/explain"
      );
      setShapData(response.data);
    } catch (error) {
      setShapError("Failed to load SHAP explanations.");
    }
  };

  fetchShapData();
}, [active]);



useEffect(() => {
  if (!streamActive) return;

  let cancelled = false;

  
const runStream = async () => {
  try {
    const response = await axios.post(
      "http://127.0.0.1:8000/api/stream/start",
      { features: Array(82).fill(0) }
    );

    if (!cancelled) {
      setStreamError("");
      setStreamResult(response.data);
      setStreamCount((count) => count + 1);
    }
  } catch (error) {
    if (!cancelled) {
      setStreamError(
        error.response?.data?.detail ||
        error.message ||
        "Streaming prediction failed."
      );
      setStreamActive(false);
    }
  }
};


  runStream();

 
  


  const intervalId = setInterval(runStream, 2000);

  return () => {
    cancelled = true;
    clearInterval(intervalId);
  };
}, [streamActive]);


  return (
    <div className="app-shell">
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) uploadDataset(file);
          event.target.value = "";
        }}
      />

      <aside className={`sidebar ${menuOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-icon"><ShieldCheck size={25} /></div>
          <div>
            <h2>Sentinel<span>AI</span></h2>
            <p>IoT SECURITY CENTER</p>
          </div>
        </div>

        <div className="nav-label">MAIN MENU</div>
        <nav>
          {navItems.map(({ name, icon: Icon }) => (
            <button
              key={name}
              className={`nav-item ${active === name ? "active" : ""}`}
              onClick={() => {
                setActive(name);
                setMenuOpen(false);
              }}
            >
              
<Icon size={19} />
<span>{name}</span>
{name === "Threat Detection" && (
  <span className="nav-count">{detectionHistory.length}</span>
)}

            </button>
          ))}
        </nav>

        <div className="nav-label tools-label">SYSTEM</div>
        <button className="nav-item" onClick={() => setActive("Settings")}>
          <Settings size={19} /><span>Settings</span>
        </button>

        <div className="sidebar-bottom">
          <ShieldCheck size={22} className="green-text" />
          <h4>System Protected</h4>
          <p>Your security dashboard is ready.</p>
          <div className="secure-status"><span /> Interface operational</div>
        </div>

        
<div className="profile">
  <div className="avatar">SA</div>
  <div className="profile-text">
    <strong>Security Admin</strong>
    <span>Administrator</span>
  </div>
  <button
    type="button"
    className="icon-button"
    onClick={() => setActive("Settings")}
    aria-label="Open Settings"
    title="Settings"
  >
    <Settings size={17} />
  </button>
</div>
</aside>


      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="mobile-menu"
              onClick={() => setMenuOpen(!menuOpen)}
            >
              <Menu />
            </button>
            <div>
              <p className="breadcrumb">
                Security Center / <strong>{active}</strong>
              </p>
              <h1>{active}</h1>
            </div>
          </div>
          
<button
  className="icon-button"
  aria-label="Notifications"
  title="Notifications"
  onClick={() => setActive("Threat Detection")}
>
  <Bell size={19} />
</button>

        </header>

        {active === "Overview" && (
          <>
            <section className="welcome-row">
              <div>
                <h2>Your network, protected.</h2>
                <p>Monitor IoT network activity and security threats.</p>
              </div>
              <div className="date-chip">
                <Activity size={16} /> ML Monitoring Preview
              </div>
            </section>

            <section className="stats-grid">
              {stats.map(({ title, value, icon: Icon }) => (
                <article className="stat-card" key={title}>
                  <div className="stat-top">
                    <div className="stat-icon"><Icon size={20} /></div>
                  </div>
                  <p>{title}</p>
                  <h3>{value}</h3>
                  <span className="stat-caption">Recorded model predictions</span>
                </article>
              ))}
            </section>

            <section className="dashboard-grid">
              <article className="panel">
                <h3>Welcome to SentinelAI</h3>
                <p>Intelligent intrusion detection for IoT networks.</p>
                <p>Use the sidebar to explore traffic, threats and AI insights.</p>
              </article>
              
<article className="panel">
  <ShieldCheck size={32} className="green-text" />
  <h3>Detection Status</h3>
  <p>Dashboard interface ready.</p>
  <p>
    ML model integration:{" "}
    {modelStatus?.model_trained === true
      ? "Connected"
      : modelStatus === null
      ? "Checking..."
      : "Unavailable"}
  </p>
  <p>
    Dataset integration:{" "}
    {uploadResult ? "Dataset uploaded" : "No dataset uploaded"}
  </p>
</article>

            </section>
            

{projectInfo && (
  <div className="dashboard-card project-info-card">
    <h3>Project Information</h3>

    <p><strong>Dataset:</strong> {projectInfo.dataset}</p>
    <p><strong>Algorithm:</strong> {projectInfo.algorithm}</p>
    <p><strong>Explainability:</strong> {projectInfo.explainability}</p>
    <p><strong>Frontend:</strong> {projectInfo.frontend}</p>
    <p><strong>Backend:</strong> {projectInfo.backend}</p>
    <p><strong>Status:</strong> {projectInfo.status}</p>
  </div>
)}


            
<div className="panel page-panel">
  <h2>System Summary</h2>
  <p>API: {systemSummary?.api_status ?? "Checking..."}</p>
  <p>Model: {systemSummary?.model_type ?? "Unknown"}</p>
  <p>Dataset: {systemSummary?.dataset ?? "Unknown"}</p>
  <p>Mode: {systemSummary?.mode ?? "Unknown"}</p>
  <p>Server: {serverTime?.status ?? "Checking..."}</p>
  <p>Last checked: {serverTime?.server_time ?? "Loading..."}</p>
  <p>
    Model trained: {systemSummary?.model_trained ? "Yes" : "No"}
  </p>
</div>

            <div className="panel page-panel">
  <h2>Recent Detection History</h2>
  {detectionHistory.map((item) => (
    <div className="detection-details" key={item.id}>
      <div>
        <span>{item.prediction} — {item.severity}</span>
        <strong>{item.risk_score}/100</strong>
      </div>
      <p>{item.source}</p>
    </div>
  ))}
  <p>History shows predictions recorded by the backend.</p>
</div>

            <AlertTable
              alerts={filteredAlerts}
              search={search}
              setSearch={setSearch}
              onView={() => setActive("Threat Detection")}
            />
          </>
        )}

        {active === "Network Traffic" && (
          <section className="panel page-panel">
  <Activity size={32} className="green-text" />
  <h2>Dataset Upload</h2>
  <p>Upload a CSV file to preview network traffic records.</p>

<button
  type="button"
  className="view-all"
  disabled={uploading}
  onClick={() => fileInputRef.current?.click()}
>
  {uploading ? "Uploading..." : "Choose CSV File"}
</button>

  {uploadError && <p>{uploadError}</p>}

  {uploadResult && (
    <>
      <p>File: {uploadResult.filename}</p>
      <p>Rows previewed: {uploadResult.rows_previewed}</p>
      <p>Columns: {uploadResult.columns.join(", ")}</p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {uploadResult.columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {uploadResult.preview.map((row, index) => (
              <tr key={index}>
                {uploadResult.columns.map((column) => (
                  <td key={column}>{String(row[column] ?? "")}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>

        {trafficAnalysis && (
          <div className="dashboard-card">
            <h3>Traffic Analysis Results</h3>
            <p>Total Traffic: {trafficAnalysis.total_traffic}</p>
            <p>Attacks Detected: {trafficAnalysis.attacks_detected}</p>
            <p>Benign Traffic: {trafficAnalysis.benign_traffic}</p>

            <h4>Attack Types</h4>
            {Object.entries(trafficAnalysis.attack_types).map(
              ([type, count]) => (
                <p key={type}>{type}: {count}</p>
              )
            )}
          </div>
        )}

        {modelPrediction && (
          <div className="dashboard-card">
            <h3>Real ML Model Predictions</h3>
            <p>Model: {modelPrediction.model}</p>
            <p>Total Rows: {modelPrediction.total_rows.toLocaleString()}</p>
            <p>
              Predicted Attacks:{" "}
              {modelPrediction.predicted_attacks.toLocaleString()}
            </p>
            <p>
              Predicted Benign:{" "}
              {modelPrediction.predicted_benign.toLocaleString()}
            </p>

            {modelPrediction.accuracy !== undefined && (
              <>
                <h4>Evaluation Metrics</h4>
                <p>Accuracy: {(modelPrediction.accuracy * 100).toFixed(2)}%</p>
                <p>Precision: {(modelPrediction.precision * 100).toFixed(2)}%</p>
                <p>Recall: {(modelPrediction.recall * 100).toFixed(2)}%</p>
                <p>F1 Score: {(modelPrediction.f1_score * 100).toFixed(2)}%</p>
                
{modelPrediction.confusion_matrix && (
  <>
    <h4>Confusion Matrix</h4>
    <table>
      <thead>
        <tr>
          <th></th>
          <th>Predicted Benign</th>
          <th>Predicted Attack</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Actual Benign</th>
          <td>{modelPrediction.confusion_matrix[0][0].toLocaleString()}</td>
          <td>{modelPrediction.confusion_matrix[0][1].toLocaleString()}</td>
        </tr>
        <tr>
          <th>Actual Attack</th>
          <td>{modelPrediction.confusion_matrix[1][0].toLocaleString()}</td>
          <td>{modelPrediction.confusion_matrix[1][1].toLocaleString()}</td>
        </tr>
      </tbody>
    </table>
  </>
)}

<div className="dashboard-card">
  <h4>Model Evaluation Chart</h4>
  
{modelPrediction.accuracy !== undefined &&
 modelPrediction.precision !== undefined &&
 modelPrediction.recall !== undefined &&
 modelPrediction.f1_score !== undefined ? (
  <ResponsiveContainer width="100%" height={250}>
    <BarChart
      data={[
        { metric: "Accuracy", score: modelPrediction.accuracy * 100 },
        { metric: "Precision", score: modelPrediction.precision * 100 },
        { metric: "Recall", score: modelPrediction.recall * 100 },
        { metric: "F1 Score", score: modelPrediction.f1_score * 100 },
      ]}
    >
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="metric" />
      <YAxis domain={[0, 100]} />
      <Tooltip />
      <Bar dataKey="score" fill="#34d399" />
    </BarChart>
  </ResponsiveContainer>
) : (
  <p>
    Evaluation metrics are unavailable. Upload a labeled CSV file
    containing a Label column to evaluate the model.
  </p>
)}

</div>


              </>
            )}
          </div>

        )}
      </div>
    </>
  )}
</section>
        )}

        
{active === "Threat Detection" && (
  <section className="page-panel">
    <h2>Threat Detection</h2>
    <p>Review and search recorded ML predictions.</p>

    
<div className="dashboard-card">
  <h3>Simulated Traffic Stream</h3>
  <p>Automatically run simulated traffic predictions.</p>

 
<div>
  <button
    type="button"
    disabled={streamActive}
    onClick={() => {
      setStreamActive(true);
      setStreamCount(0);
    }}
  >
    Start Stream
  </button>

  <button
    type="button"
    disabled={!streamActive}
    onClick={() => setStreamActive(false)}
  >
    Stop Stream
  </button>
</div>


<div className="csv-stream-controls">
  <h4>CSV Batch Stream</h4>
  <p>Process a CIC-DDoS2019 CSV file in batches of 1,000 rows.</p>

  <input
    type="file"
    accept=".csv,text/csv"
    disabled={csvStreamLoading}
    onChange={(event) => {
      const file = event.target.files?.[0];
      if (file) runCsvStream(file);
      event.target.value = "";
    }}
  />

  {csvStreamLoading && (
    <p role="status">Processing CSV batches. Please wait...</p>
  )}

  {csvStreamError && (
    <p className="stream-error" role="alert">
      {String(csvStreamError)}
    </p>
  )}

  {csvStreamResult && (
    <div className="stream-result">
      <h4>CSV Processing Results</h4>
      <p>File: {csvStreamResult.filename}</p>
      <p>Total rows: {csvStreamResult.total_rows?.toLocaleString()}</p>
      <p>Attacks detected: {csvStreamResult.attacks_detected?.toLocaleString()}</p>
      <p>Benign traffic: {csvStreamResult.benign_traffic?.toLocaleString()}</p>
      <p>Batch size: {csvStreamResult.batch_size?.toLocaleString()}</p>
      <p>Batches processed: {csvStreamResult.batches?.length ?? 0}</p>
      <p>Status: {csvStreamResult.status}</p>
    </div>
  )}
</div>



<button
  type="button"
  disabled={streamActive}
  onClick={() => {
    setStreamResult(null);
    setStreamError("");
    setStreamCount(0);
  }}
>
  Clear Stream Results
</button>


<div className="stream-status">
  <p>
    Stream Status:{" "}
    <strong>{streamActive ? "Running" : "Stopped"}</strong>
  </p>
  <p>
    Records Processed: <strong>{streamCount}</strong>
  </p>
</div>

{streamError && (
  <div className="stream-error" role="alert">
    <strong>Stream interrupted</strong>
    <p>{String(streamError)}</p>
    <button
      type="button"
      onClick={() => {
        setStreamError("");
        setStreamActive(true);
      }}
    >
      Restart Stream
    </button>
  </div>
)}

<div className="stream-connection">
  <span>
    Backend: {apiStatus === "healthy" ? "Connected" : "Disconnected"}
  </span>
  <span>
    ML Model: {modelStatus?.model_trained ? "Ready" : "Unavailable"}
  </span>
</div>

<div className="stream-summary">
  <h4>Stream Summary</h4>
  <p>Records processed: {streamCount}</p>
  <p>
    Latest prediction: {streamResult?.prediction ?? "No prediction yet"}
  </p>
  <p>
  Latest risk score:{" "}
  {streamResult?.risk_score != null
  ? `${streamResult.risk_score}%`
  : streamResult?.latest_risk_score != null
    ? `${streamResult.latest_risk_score}%`
    : "—"}
</p>
</div>




  {streamError && <p role="alert">{String(streamError)}</p>}

  {streamResult && (
    <div className="stream-result">
      <h4>Latest Stream Prediction</h4>
      <p>Prediction: <strong>{streamResult.prediction}</strong></p>
      <p>Risk Score: <strong>{streamResult.risk_score}%</strong></p>
      <p>Severity: <strong>{streamResult.severity}</strong></p>
      <p>Status: <strong>{streamResult.status}</strong></p>
    </div>
  )}
</div>


    <AlertTable
      alerts={filteredAlerts}
      search={search}
      setSearch={setSearch}
      onView={() => setActive("AI Insights")}
    />
  </section>
)}


        {active === "AI Insights" && (
          <>
            <section className="welcome-row">
              <div>
                <h2>Understand every prediction.</h2>
                <p>Explore model explanations and threat risk scores.</p>
              </div>
              <div className="date-chip">
                <BrainCircuit size={16} /> AI Workspace
              </div>
            </section>
<section className="stats-grid">
  <article className="stat-card">
    <div className="stat-icon"><BrainCircuit /></div>
    <p>ML Model</p>
    <h3>Random Forest</h3>
    <span className="stat-caption">Real CIC-DDoS2019 model</span>
  </article>

  <article className="stat-card">
    <div className="stat-icon"><ShieldAlert /></div>
    <p>Risk Assessment</p>
    <h3>Prototype</h3>
    <span className="stat-caption">Risk scoring module</span>
  </article>

  <article className="stat-card">
    <div className="stat-icon"><Activity /></div>
    <p>Explainable AI</p>
    <h3>SHAP</h3>
    <span className="stat-caption">Top 15 feature impacts</span>
  </article>

  <article className="stat-card">
    <div className="stat-icon"><CircleCheck /></div>
    <p>Model Evaluation</p>
    <h3>Implemented</h3>
    <span className="stat-caption">CSV evaluation metrics</span>
  </article>
</section>
<section className="panel">
  <div className="panel-header">
    <div>
      <h3>SHAP Feature Impact</h3>
      <p>Features influencing the model's attack prediction</p>
    </div>
  </div>

  {shapError && <p className="error-message">{shapError}</p>}

  {!shapData && !shapError && <p>Loading SHAP explanations...</p>}

  {shapData && (
    <>
      <p className="stat-caption">
        Model: {shapData.model} | Explained class: {shapData.explained_class}
      </p>

      <div style={{ width: "100%", height: 440 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={shapData?.explanation || []}
            layout="vertical"
            margin={{ top: 10, right: 25, left: 30, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis
              type="category"
              dataKey="feature"
              width={190}
              tick={{ fontSize: 11 }}
            />
            <Tooltip />
            <Bar
  dataKey="impact"
  name="SHAP impact"
  fill="#38bdf8"
  label={{ position: "right", formatter: (value) => value.toFixed(4) }}
/>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="stat-caption">{shapData.note}</p>
    </>
  )}
</section>

            <section className="panel page-panel">
  <BrainCircuit size={32} className="green-text" />
  <h2>Prediction Explanation</h2>
  <button
  className="view-all"
  onClick={async () => {
    try {
      const response = await axios.get(
        "http://127.0.0.1:8000/api/model/explain"
      );
      setExplanation(response.data);
    } catch (error) {
      alert(error.response?.data?.detail || "Explanation failed.");
    }
  }}
>
  Explain Prediction (SHAP)
</button>
<button
  className="view-all"
  onClick={async () => {
    try {
      const response = await axios.get(
        "http://127.0.0.1:8000/api/model/feature-importance"
      );
      setFeatureImportance(response.data.feature_importance);
    } catch (error) {
      alert(error.response?.data?.detail || "Failed to load feature importance.");
    }
  }}
>
  View Feature Importance
</button>


{featureImportance && featureImportance.slice(0, 15).map((item) => (
  <div className="detection-details" key={item.feature}>
    <div>
      <span>{item.feature}</span>
      <strong>{Number(item.importance).toFixed(6)}</strong>
    </div>
  </div>
))}


{explanation && (
  <div className="detection-details">
    {explanation.explanation.map((item) => (
      <div key={item.feature}>
        <span>{item.feature}</span>
        <strong>{item.impact}</strong>
      </div>
    ))}
    <p>{explanation.note}</p>
  </div>
)}

<p>
  Model status: {modelStatus?.model_trained
    ? "Real Random Forest model available"
    : "Real model not found"}
</p>

<p>
  Trained: {modelStatus?.model_trained ? "Yes" : "No"}
</p>

<p>
  Dataset: {modelStatus?.dataset ?? "Unknown"}
</p>

<p>Test the connection to the prediction API.</p>


  <button
    className="view-all"
    
onClick={async () => {
  try {
    const response = await axios.post(
      "http://127.0.0.1:8000/api/predict",
      { features: Array(82).fill(0) }
    );
    setPrediction(response.data);
    alert("Prediction API connected successfully.");
  } catch (error) {
    console.error("Prediction API error:", error.response?.data || error.message);
    alert(
      `Prediction API failed: ${
        error.response?.data?.detail || error.message
      }`
    );
  }
}}

  >
    Test Prediction API
  </button>
  
<button
  type="button"
  className="view-all"
  onClick={() => fileInputRef.current?.click()}
>
  Run ML Prediction
</button>


{riskResult && (
  <div className="detection-details">
    <div>
      <span>Prototype Risk Score</span>
      <strong>{riskResult.risk_score}/100</strong>
    </div>
    <div>
      <span>Severity</span>
      <strong>{riskResult.severity}</strong>
    </div>
    <p>{riskResult.note}</p>
  </div>
)}
  {prediction && (
  <div className="detection-details">
    <div><span>Prediction</span><strong>{prediction.prediction}</strong></div>
    <div><span>Risk Score</span><strong>{prediction.risk_score}</strong></div>
    <div><span>Severity</span><strong>{prediction.severity}</strong></div>
    <p>{prediction.message}</p>
  </div>
)}

  <p>
    Model status: Real Random Forest model available.
  </p>
 <button
  className="view-all"
  onClick={async () => {
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/api/model/train-sample"
      );
      setTrainingResult(response.data);
    } catch (error) {
      alert(error.response?.data?.detail || "Training failed.");
    }
  }}
>
  Train Sample Model
</button>
{trainingResult && (
  <div className="detection-details">
    <div><span>Accuracy</span><strong>{trainingResult.accuracy}%</strong></div>
    <div><span>Precision</span><strong>{trainingResult.precision}%</strong></div>
    <div><span>Recall</span><strong>{trainingResult.recall}%</strong></div>
    <div><span>F1 Score</span><strong>{trainingResult.f1_score}%</strong></div>
    <div>
  <div>
  <span>Confusion Matrix (TN, FP / FN, TP)</span>
  <strong>
    {trainingResult.confusion_matrix
      .map((row) => `[${row.join(", ")}]`)
      .join(" ")}
  </strong>
</div>
  <strong>
    {trainingResult.confusion_matrix
      .map((row) => row.join(" "))
      .join(" | ")}
  </strong>
</div>
    <p>{trainingResult.warning}</p>
  </div>
)}
</section>

          </>
        )}

        {active === "Settings" && (
          <section className="panel page-panel">
            <Settings size={32} className="green-text" />
            <h2>Settings</h2>
            <p>Project configuration</p>
            <div className="detection-details">
              <div><span>Application</span><strong>SentinelAI</strong></div>
              <div><span>Environment</span><strong>Development</strong></div>
              
<div>
  <span>Backend API</span>
  <strong>
    {apiStatus === "healthy" ? "Connected" :
     apiStatus === "Disconnected" ? "Disconnected" : "Checking..."}
  </strong>
</div>
<div>
  <span>Machine learning</span>
  <strong>
    {modelStatus?.model_trained === true
      ? "Connected"
      : modelStatus === null
      ? "Checking..."
      : "Model unavailable"}
  </strong>
</div>

            </div>
          </section>
        )}

        <footer className="footer">
          <span>SentinelAI <b>·</b> IoT Intrusion Detection System</span>
          <span>Demo interface · No live monitoring</span>
        </footer>
      </main>
    </div>
  );
}

function AlertTable({ alerts, search, setSearch, onView }) {
  return (
    <section className="panel alerts-panel">
      <div className="panel-heading alerts-heading">
        <div>
          <h3>Recent Threat Alerts</h3>
          <p>Real predictions recorded by the ML backend</p>
        </div>

        <div className="alert-tools">
          <div className="search-box">
            <Search size={16} />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search alerts..."
            />
          </div>

          <button className="view-all" onClick={onView}>
            View details <ArrowUpRight size={15} />
          </button>
        </div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Prediction</th>
              <th>Risk Score</th>
              <th>Severity</th>
              <th>Time</th>
              <th>Source</th>
            </tr>
          </thead>

          <tbody>
            {alerts.map((alert) => (
              <tr key={alert.id}>
                <td>{alert.prediction}</td>
                <td>{alert.risk_score}%</td>
                <td>
                  <span
                    className={`severity ${String(
                      alert.severity
                    ).toLowerCase()}`}
                  >
                    {alert.severity}
                  </span>
                </td>
                <td className="muted">
                  {alert.timestamp
                    ? new Date(alert.timestamp).toLocaleString()
                    : "Unknown"}
                </td>
                <td>{alert.source}</td>
              </tr>
            ))}

            {alerts.length === 0 && (
              <tr>
                <td colSpan="5" className="empty-state">
                  No matching detections.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="panel-footer">
        Predictions recorded during the current backend session.
      </div>
    </section>
  );
}

export default App;