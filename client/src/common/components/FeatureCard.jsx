import React from "react";
import { Card, CardActionArea, CardContent, Chip, Stack, Typography } from "@mui/material";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import "../styles/feature-card.css";

export default function FeatureCard({ icon, title, description, status, to }) {
  const content = (
    <CardContent>
      <Stack spacing={2}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={2}>
          <span className="feature-icon">{icon}</span>
          <Chip label={status || "Ready"} color="primary" variant="outlined" />
        </Stack>
        <Typography variant="h6">{title}</Typography>
        <Typography color="text.secondary">{description}</Typography>
      </Stack>
    </CardContent>
  );

  return (
    <Card component={motion.div} whileHover={{ y: -6 }} transition={{ type: "spring", stiffness: 260, damping: 18 }} className="feature-card">
      {to ? (
        <CardActionArea component={Link} to={to} className="feature-card-action">
          {content}
        </CardActionArea>
      ) : (
        content
      )}
    </Card>
  );
}
