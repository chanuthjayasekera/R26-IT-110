import React from "react"
import { Button, Card, CardContent, Container, Stack, Typography } from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import { Link } from "react-router-dom";
import "../styles/auth.css";

export default function Logout() {
  return (
    <Container maxWidth="sm" className="page auth-page">
      <Card>
        <CardContent>
          <Stack spacing={2.5} alignItems="center" textAlign="center">
            <div className="auth-icon success"><CheckCircleIcon /></div>
            <Typography variant="h4">Logged out safely</Typography>
            <Typography color="text.secondary">Your secure session has ended.</Typography>
            <Button component={Link} to="/login" variant="contained" size="large">Login again</Button>
            <Button component={Link} to="/" variant="text">Back to home</Button>
          </Stack>
        </CardContent>
      </Card>
    </Container>
  );
}
