// =========================================================================
// main.jsx — App composition + mount.
// =========================================================================

function App() {
  const [scenarioId, setScenarioId] = React.useState("real_pokhara");

  // Smooth-scroll to lab on load.
  const loadScenario = React.useCallback((id) => {
    setScenarioId(id);
    setTimeout(() => {
      const el = document.getElementById("lab");
      if (el) el.scrollIntoView ? window.scrollTo({ top: el.offsetTop - 60, behavior: "smooth" }) : null;
    }, 60);
  }, []);

  return (
    <React.Fragment>
      <window.HeroSection />
      <window.AbstractSection />
      <window.ProblemSection />
      <window.ArchitectureSection />
      <window.ScenarioAtlas loadedId={scenarioId} onLoad={loadScenario} />
      <window.LiveLab scenarioId={scenarioId} onLoad={loadScenario} />
      <window.ResultsDashboard />
      <window.GeneralizationSection />
      <window.AblationsSection />
      <window.MethodsSection />
      <window.ReproduceSection />
    </React.Fragment>
  );
}

const root = ReactDOM.createRoot(document.getElementById("app"));
root.render(<App />);
