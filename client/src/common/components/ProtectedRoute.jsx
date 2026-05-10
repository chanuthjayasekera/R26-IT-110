import React from "react";
import { CircularProgress, Stack, Typography } from "@mui/material";
import { Navigate } from "react-router-dom";
import { useAuth } from "../state/AuthContext.jsx";

function homeForRole(role) {
  if (role === "admin") return "/admin";
  if (role === "professional") return "/professional";
  return "/patient";
}

export default function ProtectedRoute({ role, children }) {
  const { user, booting } = useAuth();

  if (booting) {
    return (
      <Stack minHeight="60vh" alignItems="center" justifyContent="center" spacing={2}>
        <CircularProgress />
        <Typography>Checking secure session...</Typography>
      </Stack>
    );
  }

  if (!user) return <Navigate to="/login" replace />;
  if (role && user.role !== role) return <Navigate to={homeForRole(user.role)} replace />;
  return children;
}
