import React from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./common/components/Layout.jsx";
import ProtectedRoute from "./common/components/ProtectedRoute.jsx";
import Home from "./common/pages/Home.jsx";
import Login from "./common/pages/Login.jsx";
import Register from "./common/pages/Register.jsx";
import ForgotPassword from "./common/pages/ForgotPassword.jsx";
import ResetPassword from "./common/pages/ResetPassword.jsx";
import PatientDashboard from "./common/pages/PatientDashboard.jsx";
import ProfessionalDashboard from "./common/pages/ProfessionalDashboard.jsx";
import AdminDashboard from "./common/pages/AdminDashboard.jsx";
import Profile from "./common/pages/Profile.jsx";
import Logout from "./common/pages/Logout.jsx";
import NormalAbnormalDetection from "./normal-abnormal-detection-system/pages/NormalAbnormalDetection.jsx";
import ScaKoaDetection from "./sca-koa-detection-system/pages/ScaKoaDetection.jsx";
import ParkinsonDetection from "./pd-detection-system/pages/ParkinsonDetection.jsx";
import RehabExerciseDetection from "./recommendation-rehabilitation-system/pages/RehabExerciseDetection.jsx";
import NormalAbnormalResults from "./normal-abnormal-detection-system/pages/NormalAbnormalResults.jsx";
import ScaKoaResults from "./sca-koa-detection-system/pages/ScaKoaResults.jsx";
import ParkinsonResults from "./pd-detection-system/pages/ParkinsonResults.jsx";
import RehabExerciseResults from "./recommendation-rehabilitation-system/pages/RehabExerciseResults.jsx";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/reset-password" element={<ResetPassword />} />
        <Route path="/logout" element={<Logout />} />
        <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
        <Route path="/patient" element={<ProtectedRoute role="patient"><PatientDashboard /></ProtectedRoute>} />
        <Route path="/patient/detection/normal-abnormal" element={<ProtectedRoute role="patient"><NormalAbnormalDetection /></ProtectedRoute>} />
        <Route path="/patient/results/normal-abnormal" element={<ProtectedRoute role="patient"><NormalAbnormalResults /></ProtectedRoute>} />
        <Route path="/patient/detection/sca-koa" element={<ProtectedRoute role="patient"><ScaKoaDetection /></ProtectedRoute>} />
        <Route path="/patient/results/sca-koa" element={<ProtectedRoute role="patient"><ScaKoaResults /></ProtectedRoute>} />
        <Route path="/patient/detection/pd" element={<ProtectedRoute role="patient"><ParkinsonDetection /></ProtectedRoute>} />
        <Route path="/patient/results/pd" element={<ProtectedRoute role="patient"><ParkinsonResults /></ProtectedRoute>} />
        <Route path="/patient/detection/rehab-exercise" element={<ProtectedRoute role="patient"><RehabExerciseDetection /></ProtectedRoute>} />
        <Route path="/patient/results/rehab-exercise" element={<ProtectedRoute role="patient"><RehabExerciseResults /></ProtectedRoute>} />
        <Route path="/patient/detection/rehab" element={<ProtectedRoute role="patient"><RehabExerciseDetection /></ProtectedRoute>} />
        <Route path="/professional" element={<ProtectedRoute role="professional"><ProfessionalDashboard /></ProtectedRoute>} />
        <Route path="/admin" element={<ProtectedRoute role="admin"><AdminDashboard /></ProtectedRoute>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
