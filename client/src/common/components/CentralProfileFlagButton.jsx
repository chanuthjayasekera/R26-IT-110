import React, { useEffect, useState } from "react";
import { Button, Tooltip } from "@mui/material";
import BeenhereIcon from "@mui/icons-material/Beenhere";
import FlagOutlinedIcon from "@mui/icons-material/FlagOutlined";
import { api, getApiError } from "../api/http.js";

export default function CentralProfileFlagButton({ sourceType, screeningId, onFlagged }) {
  const [selectedId, setSelectedId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    setError("");
    if (!sourceType || !screeningId) return undefined;

    api.get("/central-profile")
      .then((res) => {
        if (!alive) return;
        const flag = (res.data.flags || []).find((item) => item.sourceType === sourceType);
        setSelectedId(flag?.screeningId || "");
      })
      .catch(() => {
        if (alive) setSelectedId("");
      });

    return () => {
      alive = false;
    };
  }, [sourceType, screeningId]);

  const selected = selectedId === screeningId;

  async function saveFlag() {
    if (selected || busy) return;
    setBusy(true);
    setError("");
    try {
      const res = await api.post("/central-profile/flags", { sourceType, screeningId });
      setSelectedId(res.data.flag?.screeningId || screeningId);
      onFlagged?.(res.data.flag);
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Unable to update centralized profile.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Tooltip title={error || (selected ? "This is the selected centralized result for this model." : "Replace the centralized result for this model with this screening.")}>
      <span>
        <Button
          variant={selected ? "contained" : "outlined"}
          color={selected ? "success" : "primary"}
          startIcon={selected ? <BeenhereIcon /> : <FlagOutlinedIcon />}
          disabled={busy || selected}
          onClick={saveFlag}
        >
          {selected ? "Central result selected" : busy ? "Saving..." : "Set as central result"}
        </Button>
      </span>
    </Tooltip>
  );
}
