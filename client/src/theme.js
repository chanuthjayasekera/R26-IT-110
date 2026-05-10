import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#0284c7", light: "#38bdf8", dark: "#075985" },
    secondary: { main: "#0f766e" },
    background: { default: "#f4fbff", paper: "#ffffff" },
    text: { primary: "#0f172a", secondary: "#475569" },
    success: { main: "#16a34a" },
    warning: { main: "#d97706" },
    error: { main: "#dc2626" }
  },
  typography: {
    fontFamily: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Arial"].join(","),
    h1: { fontWeight: 800 },
    h2: { fontWeight: 800 },
    h3: { fontWeight: 800 },
    h4: { fontWeight: 800 },
    h5: { fontWeight: 800 },
    h6: { fontWeight: 700 },
    button: { textTransform: "none", fontWeight: 700 }
  },
  shape: { borderRadius: 18 },
  components: {
    MuiButton: { styleOverrides: { root: { borderRadius: 14, paddingInline: 22, paddingBlock: 10 } } },
    MuiTextField: { defaultProps: { fullWidth: true, variant: "outlined" } },
    MuiCard: { styleOverrides: { root: { borderRadius: 28, boxShadow: "0 22px 60px rgba(2, 132, 199, 0.12)" } } }
  }
});
