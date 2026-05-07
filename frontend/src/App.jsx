import { Routes, Route, Navigate } from "react-router-dom";
import Nav from "./components/Nav.jsx";
import Demo from "./pages/Demo.jsx";
import Metrics from "./pages/Metrics.jsx";

export default function App() {
  return (
    <>
      <Nav />
      <Routes>
        <Route path="/" element={<Navigate to="/demo" replace />} />
        <Route path="/demo" element={<Demo />} />
        <Route path="/metrics" element={<Metrics />} />
      </Routes>
    </>
  );
}
