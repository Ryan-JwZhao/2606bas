(function exposeStreamCoordinates(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.BASStreamCoordinates = api;
  }
}(typeof globalThis !== 'undefined' ? globalThis : this, () => {
  function finitePositive(value) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? number : null;
  }

  function clampFrameCoordinate(value, size) {
    return Math.max(0, Math.min(size - 1, Math.round(value)));
  }

  function mapRenderedPointToFrame({
    clientX,
    clientY,
    rect,
    frameWidth,
    frameHeight,
    objectFit = 'fill',
    positionX = 0.5,
    positionY = 0.5,
  }) {
    const width = finitePositive(frameWidth);
    const height = finitePositive(frameHeight);
    const rectWidth = finitePositive(rect?.width);
    const rectHeight = finitePositive(rect?.height);
    const pointX = Number(clientX);
    const pointY = Number(clientY);
    if (!width || !height || !rectWidth || !rectHeight || !Number.isFinite(pointX) || !Number.isFinite(pointY)) {
      return null;
    }

    const rectLeft = Number(rect?.left) || 0;
    const rectTop = Number(rect?.top) || 0;
    const fit = String(objectFit || 'fill').trim().toLowerCase();
    if (fit !== 'cover' && fit !== 'contain' && fit !== 'scale-down') {
      const sourceX = (pointX - rectLeft) * width / rectWidth;
      const sourceY = (pointY - rectTop) * height / rectHeight;
      if (sourceX < 0 || sourceX > width || sourceY < 0 || sourceY > height) return null;
      return {
        x: clampFrameCoordinate(sourceX, width),
        y: clampFrameCoordinate(sourceY, height),
      };
    }

    const containScale = Math.min(rectWidth / width, rectHeight / height);
    const scale = fit === 'cover'
      ? Math.max(rectWidth / width, rectHeight / height)
      : (fit === 'scale-down' ? Math.min(1, containScale) : containScale);
    const renderedWidth = width * scale;
    const renderedHeight = height * scale;
    const horizontalPosition = Math.max(0, Math.min(1, Number(positionX)));
    const verticalPosition = Math.max(0, Math.min(1, Number(positionY)));
    const contentLeft = rectLeft + (rectWidth - renderedWidth) * horizontalPosition;
    const contentTop = rectTop + (rectHeight - renderedHeight) * verticalPosition;
    const sourceX = (pointX - contentLeft) / scale;
    const sourceY = (pointY - contentTop) / scale;
    if (sourceX < 0 || sourceX > width || sourceY < 0 || sourceY > height) return null;
    return {
      x: clampFrameCoordinate(sourceX, width),
      y: clampFrameCoordinate(sourceY, height),
    };
  }

  return { mapRenderedPointToFrame };
}));
