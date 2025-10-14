import React from 'react';
import { createPortal } from 'react-dom';

export default function Overlays({ hoverUi, autoOverlays, cameraOverlay }) {
  const wrapperStyle = {
    position: 'fixed',
    left: 0,
    top: 0,
    width: '100vw',
    height: '100vh',
    pointerEvents: 'none',
    zIndex: 999999,
  };

  return createPortal(
    <div style={wrapperStyle}>
      {/* hover tooltip disabled */}
    </div>,
    document.body
  );
}
