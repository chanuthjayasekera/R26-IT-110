import React, { useEffect, useMemo, useState } from "react";
import { Box, Button, Card, CardContent, Chip, Container, Grid2, Stack, Typography } from "@mui/material";
import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import AccessibilityNewIcon from "@mui/icons-material/AccessibilityNew";
import LocalHospitalIcon from "@mui/icons-material/LocalHospital";
import TimelineIcon from "@mui/icons-material/Timeline";
import HealthAndSafetyIcon from "@mui/icons-material/HealthAndSafety";
import PsychologyIcon from "@mui/icons-material/Psychology";
import FitnessCenterIcon from "@mui/icons-material/FitnessCenter";
import FeatureCard from "../components/FeatureCard.jsx";
import gaitImage from "../../assets/gait2.png";
import gaitScreeningImage from "../../assets/gait-screening.png";
import gaitRehabImage from "../../assets/gait-rehab.png";
import gaitClinicalImage from "../../assets/gait-clinical.png";
import "../styles/home.css";

export default function Home() {
  const slides = useMemo(
    () => [
      {
        image: gaitImage,
        label: "Severity-aware tracking",
        badge: "Camera-only gait screening platform",
        title: "Detect movement risk early, track recovery clearly.",
        quote: "A patient’s walking pattern can become a timeline of neurological and musculoskeletal health.",
        description:
          "Secure dashboards for normal vs abnormal screening, SCA and KOA detection, Parkinson detection, clinical risk visibility, and rehabilitation support.",
        icon: <TimelineIcon />
      },
      {
        image: gaitScreeningImage,
        label: "Multi-disorder screening",
        badge: "AI gait intelligence ecosystem",
        title: "Turn walking videos into clinical insight.",
        quote: "Small gait changes can reveal important movement patterns before they become obvious.",
        description:
          "Guide patients through abnormality screening, disorder-specific detection, severity tracking, and longitudinal profile review.",
        icon: <PsychologyIcon />
      },
      {
        image: gaitRehabImage,
        label: "Rehabilitation support",
        badge: "Patient-centered recovery monitoring",
        title: "Support rehab progress with clearer feedback.",
        quote: "Recovery becomes easier to understand when movement quality is measured over time.",
        description:
          "Patients can continue from screening into rehabilitation guidance while clinicians manage risks and recommendations.",
        icon: <FitnessCenterIcon />
      }
    ],
    []
  );

  const [activeIndex, setActiveIndex] = useState(0);
  const activeSlide = slides[activeIndex];

  useEffect(() => {
    const timer = setInterval(() => {
      setActiveIndex((current) => (current + 1) % slides.length);
    }, 4200);

    return () => clearInterval(timer);
  }, [slides.length]);

  return (
    <Container maxWidth="xl" className="page home-page">
      <Grid2 container spacing={4} alignItems="flex-start">
        <Grid2 size={{ xs: 12, md: 6 }} className="hero-left">
          <AnimatePresence mode="wait">
            <Stack
              key={activeIndex}
              spacing={1.5}
              component={motion.div}
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -14 }}
              transition={{ duration: 0.35 }}
            >
              <Typography variant="overline" color="primary" fontWeight={900}>
                {activeSlide.badge}
              </Typography>

              <Typography variant="h2" className="hero-title">
                {activeSlide.title}
              </Typography>

              <Typography variant="h6" color="text.secondary" className="hero-quote">
                “{activeSlide.quote}”
              </Typography>

              <Typography color="text.secondary" className="hero-description">
                {activeSlide.description}
              </Typography>

              <Stack direction={{ xs: "column", sm: "row" }} spacing={2} className="hero-actions">
                <Button component={Link} to="/register" size="large" variant="contained">
                  Create account
                </Button>
                <Button component={Link} to="/login" size="large" variant="outlined">
                  Login
                </Button>
              </Stack>

              <Stack direction="row" spacing={1} className="hero-tabs">
                {slides.map((slide, index) => (
                  <Chip
                    key={slide.label}
                    icon={slide.icon}
                    label={slide.label}
                    clickable
                    className={activeIndex === index ? "hero-tab active" : "hero-tab"}
                    onClick={() => setActiveIndex(index)}
                  />
                ))}
              </Stack>
            </Stack>
          </AnimatePresence>
        </Grid2>

        <Grid2 size={{ xs: 12, md: 6 }}>
          <Card className="hero-card">
            <CardContent>
              <Box className="hero-visual">
                <AnimatePresence mode="wait">
                  <motion.img
                    key={activeIndex}
                    src={activeSlide.image}
                    alt="AI gait analysis and walking pattern detection"
                    initial={{ opacity: 0, scale: 1.04 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.98 }}
                    transition={{ duration: 0.45 }}
                  />
                </AnimatePresence>

                <Box className="floating-panel panel-a">
                  {activeSlide.icon}
                  <Typography fontWeight={800}>{activeSlide.label}</Typography>
                </Box>

                <Box className="floating-panel panel-b">
                  <HealthAndSafetyIcon />
                  <Typography fontWeight={800}>Clinical oversight</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid2>
      </Grid2>

      <Grid2 container spacing={3} mt={4}>
        <Grid2 size={{ xs: 12, md: 4 }}>
          <FeatureCard icon={<AccessibilityNewIcon />} title="Patient pathway" description="Upload gait videos, view screening modules, check biometrics, and continue rehab." />
        </Grid2>
        <Grid2 size={{ xs: 12, md: 4 }}>
          <FeatureCard icon={<LocalHospitalIcon />} title="Verified clinical control" description="Medical professional accounts require proof, hospital email validation, and admin approval." />
        </Grid2>
        <Grid2 size={{ xs: 12, md: 4 }}>
          <FeatureCard icon={<TimelineIcon />} title="Longitudinal profile" description="Designed for gait history, severity progression, instability maps, and rehab outcomes." />
        </Grid2>
      </Grid2>

      <Grid2 container spacing={3} className="home-system-section">
        <Grid2 size={{ xs: 12, md: 3 }}>
          <Box className="system-step">
            <Typography fontWeight={900}>01</Typography>
            <Typography variant="h6" fontWeight={900}>Screen</Typography>
            <Typography color="text.secondary">Normal vs abnormal gait detection.</Typography>
          </Box>
        </Grid2>

        <Grid2 size={{ xs: 12, md: 3 }}>
          <Box className="system-step">
            <Typography fontWeight={900}>02</Typography>
            <Typography variant="h6" fontWeight={900}>Classify</Typography>
            <Typography color="text.secondary">SCA, KOA, and Parkinson detection paths.</Typography>
          </Box>
        </Grid2>

        <Grid2 size={{ xs: 12, md: 3 }}>
          <Box className="system-step">
            <Typography fontWeight={900}>03</Typography>
            <Typography variant="h6" fontWeight={900}>Interpret</Typography>
            <Typography color="text.secondary">Severity, risk visibility, and clinical profile.</Typography>
          </Box>
        </Grid2>

        <Grid2 size={{ xs: 12, md: 3 }}>
          <Box className="system-step">
            <Typography fontWeight={900}>04</Typography>
            <Typography variant="h6" fontWeight={900}>Recover</Typography>
            <Typography color="text.secondary">Rehabilitation support and recommendation flow.</Typography>
          </Box>
        </Grid2>
      </Grid2>

      <Grid2 container spacing={3} className="home-showcase-section">
        <Grid2 size={{ xs: 12, md: 5 }}>
          <Box className="showcase-copy">
            <Typography variant="overline" color="primary" fontWeight={900}>
              Built for patient monitoring and clinical review
            </Typography>
            <Typography variant="h4" fontWeight={900}>
              One platform for screening, risk visibility, and rehabilitation progress.
            </Typography>
            <Typography color="text.secondary">
              The system is designed to support camera-based gait analysis, clinician-approved medical professional access, longitudinal profile tracking, Excellency of AI to predict disorders early.
            </Typography>
            <Button component={Link} to="/register" variant="contained" size="large">
              Start secure registration
            </Button>
          </Box>
        </Grid2>

        <Grid2 size={{ xs: 12, md: 7 }}>
          <Box className="showcase-image-card">
            <img src={gaitClinicalImage} alt="Clinical gait analysis dashboard preview" />
            <Box className="showcase-badge">
              <TimelineIcon />
              <Typography fontWeight={900}>Central gait profile ready</Typography>
            </Box>
          </Box>
        </Grid2>
      </Grid2>
    </Container>
  );
}