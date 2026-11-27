import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, RequireAuth } from "./auth/AuthContext";
import { Layout } from "./components/Layout";
import { HomePage } from "./pages/HomePage";
import { LoginPage } from "./pages/LoginPage";
import { OutbreaksPage } from "./pages/OutbreaksPage";
import { PlantPage } from "./pages/PlantPage";
import { PlantsPage } from "./pages/PlantsPage";
import { ResultPage } from "./pages/ResultPage";
import { ScanPage } from "./pages/ScanPage";
import { SeverityPage } from "./pages/SeverityPage";

// Route map. It follows the user-flow diagram:
//   Login / Sign up -> Home -> Disease detection | My plants | Local outbreaks
export function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/login" element={<LoginPage />} />
            <Route element={<RequireAuth />}>
              <Route path="/" element={<HomePage />} />
              <Route path="/scan" element={<ScanPage />} />
              <Route path="/scans/:id" element={<ResultPage />} />
              <Route path="/scans/:id/severity" element={<SeverityPage />} />
              <Route path="/plants" element={<PlantsPage />} />
              <Route path="/plants/:id" element={<PlantPage />} />
              <Route path="/outbreaks" element={<OutbreaksPage />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
