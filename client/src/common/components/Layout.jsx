import React, { useState } from "react";
import { Avatar, Box, Button, Container, Divider, IconButton, Menu, MenuItem, Stack, Typography } from "@mui/material";
import { Link, useNavigate } from "react-router-dom";
import MonitorHeartIcon from "@mui/icons-material/MonitorHeart";
import LogoutIcon from "@mui/icons-material/Logout";
import AccountCircleIcon from "@mui/icons-material/AccountCircle";
import { useAuth } from "../state/AuthContext.jsx";
import "../styles/layout.css";

function initials(name = "") {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "US";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
}

function navFor(user) {
  if (!user) return [];
  if (user.role === "patient") {
    return [
      ["Normal vs Abnormal Detection", "/patient/detection/normal-abnormal"],
      ["SCA and KOA Detection", "/patient/detection/sca-koa"],
      ["PD Detection", "/patient/detection/pd"],
      ["Rehab Exercise Detection", "/patient/detection/rehab-exercise"]
    ];
  }
  if (user.role === "professional") {
    return [
      ["Upload Risks", "/professional"],
      ["Upload Recommendations", "/professional"],
      ["Upload Rehabilitation", "/professional"],
      ["Engage Patient Profiles", "/professional"]
    ];
  }
  return [
    ["Patient Management", "/admin"],
    ["Dr Management", "/admin"]
  ];
}

export default function Layout({ children }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [anchor, setAnchor] = useState(null);

  async function onLogout() {
    await logout();
    setAnchor(null);
    navigate("/logout");
  }

  return (
    <Box className="app-shell">
      <Box component="header" className="site-header">
        <Container maxWidth="xl">
          <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={2}>
            <Stack component={Link} to="/" direction="row" alignItems="center" spacing={1.4} className="brand-link">
              <Box className="brand-icon"><MonitorHeartIcon /></Box>
              <Box>
                <Typography fontWeight={900} lineHeight={1}>GaitAI Care</Typography>
                <Typography variant="caption" color="text.secondary">Clinical gait intelligence</Typography>
              </Box>
            </Stack>
            <Stack direction="row" spacing={0.5} alignItems="center" className="nav-actions">
              {user ? (
                <>
                  <Stack direction="row" spacing={0.5} className="desktop-nav">
                    {navFor(user).map(([label, to]) => (
                      <Button key={label} component={Link} to={to} size="small">{label}</Button>
                    ))}
                  </Stack>
                  <IconButton onClick={(event) => setAnchor(event.currentTarget)} className="profile-trigger">
                    <Avatar src={user.profileImage || ""}>{initials(user.fullName)}</Avatar>
                  </IconButton>
                  <Menu anchorEl={anchor} open={Boolean(anchor)} onClose={() => setAnchor(null)}>
                    <Box px={2} py={1}>
                      <Typography fontWeight={900}>{user.fullName}</Typography>
                      <Typography variant="caption" color="text.secondary">{user.email}</Typography>
                    </Box>
                    <Divider />
                    <MenuItem onClick={() => { setAnchor(null); navigate("/profile"); }}>
                      <AccountCircleIcon fontSize="small" /> Profile & settings
                    </MenuItem>
                    <MenuItem onClick={onLogout}>
                      <LogoutIcon fontSize="small" /> Logout
                    </MenuItem>
                  </Menu>
                </>
              ) : (
                <>
                  <Button component={Link} to="/login">Login</Button>
                  <Button component={Link} to="/register" variant="contained">Register</Button>
                </>
              )}
            </Stack>
          </Stack>
        </Container>
      </Box>
      <Box component="main" className="main-content">{children}</Box>
      <Box component="footer" className="site-footer">
        <Box className="footer-top">
          <Container maxWidth="xl">
            <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" spacing={2}>
              <Typography fontWeight={800}>Get connected with us for clinical gait intelligence updates.</Typography>
              <Stack direction="row" spacing={2} className="social-links">
                <span>f</span><span>x</span><span>in</span><span>gh</span>
              </Stack>
            </Stack>
          </Container>
        </Box>
        <Container maxWidth="xl">
          <Box className="footer-grid">
            <Box>
              <Typography variant="overline" fontWeight={900}>Company</Typography>
              <Typography variant="body2">GaitAI Care supports camera-only gait screening, clinical visibility, and rehabilitation tracking.</Typography>
            </Box>
            <Box>
              <Typography variant="overline" fontWeight={900}>Patient Modules</Typography>
              <Typography variant="body2">Normal vs abnormal detection</Typography>
              <Typography variant="body2">SCA, KOA, and PD analysis</Typography>
              <Typography variant="body2">Rehab & recommendation support</Typography>
            </Box>
            <Box>
              <Typography variant="overline" fontWeight={900}>Professional Tools</Typography>
              <Typography variant="body2">Upload risks</Typography>
              <Typography variant="body2">Upload recommendations</Typography>
              <Typography variant="body2">Engage patient profiles</Typography>
            </Box>
            <Box>
              <Typography variant="overline" fontWeight={900}>Contact</Typography>
              <Typography variant="body2">Clinical Research Unit</Typography>
              <Typography variant="body2">support@gaitai.local</Typography>
              <Typography variant="body2">+94 000 000 000</Typography>
            </Box>
          </Box>
          <Typography variant="body2" textAlign="center" className="footer-copy">© 2026 GaitAI Care. Decision-support platform.</Typography>
        </Container>
      </Box>
    </Box>
  );
}
