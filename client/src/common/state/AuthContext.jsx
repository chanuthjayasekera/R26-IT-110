import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { api, getApiError } from "../api/http.js";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [booting, setBooting] = useState(true);

  async function refreshUser() {
    try {
      const res = await api.get("/auth/me");
      setUser(res.data.user);
    } catch {
      setUser(null);
    } finally {
      setBooting(false);
    }
  }

  async function register(payload) {
    try {
      const res = await api.post("/auth/register", payload);
      setUser(res.data.user);
      return res.data;
    } catch (error) {
      throw getApiError(error);
    }
  }

  async function login(payload) {
    try {
      const res = await api.post("/auth/login", payload);
      setUser(res.data.user);
      return res.data;
    } catch (error) {
      throw getApiError(error);
    }
  }

  async function updateProfile(payload) {
    try {
      const res = await api.put("/auth/profile", payload);
      setUser(res.data.user);
      return res.data;
    } catch (error) {
      throw getApiError(error);
    }
  }

  async function logout() {
    try {
      await api.post("/auth/logout");
    } finally {
      setUser(null);
    }
  }

  useEffect(() => {
    refreshUser();
  }, []);

  const value = useMemo(() => ({ user, booting, register, login, logout, refreshUser, updateProfile }), [user, booting]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
