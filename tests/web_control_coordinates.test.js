const assert = require('node:assert/strict');
const test = require('node:test');

const { mapRenderedPointToFrame } = require('../bas/web_control/stream_coordinates.js');

function renderedPoint({ source, frame, rect, objectFit }) {
  const scale = objectFit === 'cover'
    ? Math.max(rect.width / frame.width, rect.height / frame.height)
    : Math.min(rect.width / frame.width, rect.height / frame.height);
  const renderedWidth = frame.width * scale;
  const renderedHeight = frame.height * scale;
  return {
    clientX: rect.left + (rect.width - renderedWidth) / 2 + source.x * scale,
    clientY: rect.top + (rect.height - renderedHeight) / 2 + source.y * scale,
  };
}

test('maps a same-aspect click directly into frame coordinates', () => {
  const rect = { left: 10, top: 20, width: 1600, height: 900 };
  const frame = { width: 1280, height: 720 };
  const point = renderedPoint({ source: { x: 200, y: 300 }, frame, rect, objectFit: 'cover' });

  assert.deepEqual(
    mapRenderedPointToFrame({ ...point, rect, frameWidth: frame.width, frameHeight: frame.height, objectFit: 'cover' }),
    { x: 200, y: 300 },
  );
});

test('compensates for vertical cropping caused by object-fit cover', () => {
  const rect = { left: 0, top: 0, width: 1600, height: 900 };
  const frame = { width: 1280, height: 1024 };
  const point = renderedPoint({ source: { x: 200, y: 200 }, frame, rect, objectFit: 'cover' });

  assert.deepEqual(
    mapRenderedPointToFrame({ ...point, rect, frameWidth: frame.width, frameHeight: frame.height, objectFit: 'cover' }),
    { x: 200, y: 200 },
  );
});

test('compensates for horizontal letterboxing caused by object-fit contain', () => {
  const rect = { left: 0, top: 0, width: 2400, height: 1080 };
  const frame = { width: 1920, height: 1080 };
  const point = renderedPoint({ source: { x: 200, y: 200 }, frame, rect, objectFit: 'contain' });

  assert.deepEqual(
    mapRenderedPointToFrame({ ...point, rect, frameWidth: frame.width, frameHeight: frame.height, objectFit: 'contain' }),
    { x: 200, y: 200 },
  );
});

test('maps contain-mode coordinates when the image is upscaled', () => {
  const rect = { left: 0, top: 0, width: 3840, height: 2160 };
  const frame = { width: 1920, height: 1080 };
  const point = renderedPoint({ source: { x: 200, y: 200 }, frame, rect, objectFit: 'contain' });

  assert.deepEqual(
    mapRenderedPointToFrame({ ...point, rect, frameWidth: frame.width, frameHeight: frame.height, objectFit: 'contain' }),
    { x: 200, y: 200 },
  );
});

test('rejects clicks in contain-mode letterbox bars', () => {
  const rect = { left: 0, top: 0, width: 2400, height: 1080 };

  assert.equal(
    mapRenderedPointToFrame({
      clientX: 100,
      clientY: 500,
      rect,
      frameWidth: 1920,
      frameHeight: 1080,
      objectFit: 'contain',
    }),
    null,
  );
});
