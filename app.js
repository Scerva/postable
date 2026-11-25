// ════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════

const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights';
const CLUSTER_THRESHOLD = 0.5;   // euclidean distance below which two faces are the same person
const FACE_CROP_SIZE = 80;        // px for face picker thumbnails

// ════════════════════════════════════════════════════════
// State
// ════════════════════════════════════════════════════════

const state = {
  photos: [],       // { file, img, dataUrl }[]
  detections: [],   // { box, landmarks, descriptor }[][]  — index matches photos[]
  faceSets: [],     // { baseFaceIdx, options: { photoIdx, faceIdx }[] }[]
  faceSetSelections: {}, // { [baseFaceIdx]: optionIndex }
  basePhotoIdx: 0,
};

// ════════════════════════════════════════════════════════
// DOM refs
// ════════════════════════════════════════════════════════

const screens = {
  upload:  document.getElementById('screen-upload'),
  loading: document.getElementById('screen-loading'),
  picker:  document.getElementById('screen-picker'),
  result:  document.getElementById('screen-result'),
};

const els = {
  fileInput:     document.getElementById('file-input'),
  dropzone:      document.getElementById('dropzone'),
  photoThumbs:   document.getElementById('photo-thumbs'),
  uploadActions: document.getElementById('upload-actions'),
  btnAnalyze:    document.getElementById('btn-analyze'),
  photoCount:    document.getElementById('photo-count'),
  loadingStatus: document.getElementById('loading-status'),
  loadingSub:    document.getElementById('loading-sub'),
  progressFill:  document.getElementById('progress-fill'),
  btnBack:       document.getElementById('btn-back'),
  baseSelector:  document.getElementById('base-selector'),
  peopleList:    document.getElementById('people-list'),
  btnCompose:    document.getElementById('btn-compose'),
  btnBackResult: document.getElementById('btn-back-result'),
  resultCanvas:  document.getElementById('result-canvas'),
  btnDownload:   document.getElementById('btn-download'),
  btnRestart:    document.getElementById('btn-restart'),
};

// ════════════════════════════════════════════════════════
// Navigation
// ════════════════════════════════════════════════════════

function showScreen(id) {
  for (const [key, el] of Object.entries(screens)) {
    el.classList.toggle('active', key === id);
  }
}

// ════════════════════════════════════════════════════════
// Photo upload
// ════════════════════════════════════════════════════════

async function handleFiles(files) {
  for (const file of Array.from(files)) {
    if (!file.type.startsWith('image/')) continue;
    const dataUrl = await readAsDataURL(file);
    const img = await loadImage(dataUrl);
    state.photos.push({ file, img, dataUrl });
  }
  renderThumbnails();
}

function readAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function renderThumbnails() {
  els.photoThumbs.innerHTML = '';
  state.photos.forEach((photo, idx) => {
    const wrap = document.createElement('div');
    wrap.className = 'photo-thumb';

    const img = document.createElement('img');
    img.src = photo.dataUrl;
    img.alt = `Photo ${idx + 1}`;

    const btn = document.createElement('button');
    btn.className = 'remove-btn';
    btn.innerHTML = '&times;';
    btn.setAttribute('aria-label', `Remove photo ${idx + 1}`);
    btn.addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();
      state.photos.splice(idx, 1);
      renderThumbnails();
    });

    wrap.appendChild(img);
    wrap.appendChild(btn);
    els.photoThumbs.appendChild(wrap);
  });

  const has = state.photos.length > 0;
  els.photoThumbs.classList.toggle('hidden', !has);
  els.uploadActions.classList.toggle('hidden', !has);
  els.photoCount.textContent = state.photos.length;
}

// ════════════════════════════════════════════════════════
// Face detection
// ════════════════════════════════════════════════════════

let modelsLoaded = false;

async function loadModels() {
  if (modelsLoaded) return;
  setProgress(5, 'Downloading face detector…');
  await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
  setProgress(42, 'Downloading landmark model…');
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  setProgress(78, 'Downloading recognition model…');
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
  setProgress(100, 'Models ready');
  modelsLoaded = true;
}

async function detectAllFaces() {
  state.detections = [];
  const opts = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.4 });

  for (let i = 0; i < state.photos.length; i++) {
    setProgress(
      Math.round(((i + 0.5) / state.photos.length) * 100),
      `Scanning photo ${i + 1} of ${state.photos.length}…`
    );
    // Yield so the browser can paint the progress update
    await new Promise(r => setTimeout(r, 20));

    const results = await faceapi
      .detectAllFaces(state.photos[i].img, opts)
      .withFaceLandmarks()
      .withFaceDescriptors();

    state.detections.push(results.map(r => ({
      box: r.detection.box,
      landmarks: r.landmarks,
      descriptor: r.descriptor,
    })));
  }
}

function setProgress(pct, label) {
  els.progressFill.style.width = pct + '%';
  els.loadingStatus.textContent = label;
}

// ════════════════════════════════════════════════════════
// Face sets
// Anchored on the base photo: for each detected face in the base photo,
// find the closest matching face in every other photo.
// This avoids cluster-ID bookkeeping and stale centroids.
// ════════════════════════════════════════════════════════

function buildFaceSets() {
  const baseFaces = state.detections[state.basePhotoIdx];
  state.faceSets = baseFaces.map((baseFace, baseFaceIdx) => {
    // Option 0 is always the base photo face itself
    const options = [{ photoIdx: state.basePhotoIdx, faceIdx: baseFaceIdx }];

    state.detections.forEach((dets, photoIdx) => {
      if (photoIdx === state.basePhotoIdx) return;
      let bestDist = CLUSTER_THRESHOLD;
      let bestFaceIdx = -1;
      dets.forEach((face, faceIdx) => {
        const dist = faceapi.euclideanDistance(baseFace.descriptor, face.descriptor);
        if (dist < bestDist) { bestDist = dist; bestFaceIdx = faceIdx; }
      });
      if (bestFaceIdx >= 0) options.push({ photoIdx, faceIdx: bestFaceIdx });
    });

    return { baseFaceIdx, options };
  });
}

function initFaceSetSelections() {
  state.faceSetSelections = {};
  for (const fs of state.faceSets) {
    state.faceSetSelections[fs.baseFaceIdx] = 0; // default: use the base photo face
  }
}

// Pick base photo = the one with the most faces detected
function pickDefaultBase() {
  state.basePhotoIdx = state.detections.reduce((best, det, idx) =>
    det.length > state.detections[best].length ? idx : best, 0
  );
}

// ════════════════════════════════════════════════════════
// Face crop helper — draws a padded crop of the face onto a small canvas
// ════════════════════════════════════════════════════════

function makeFaceCropCanvas(photoIdx, faceIdx) {
  const { img } = state.photos[photoIdx];
  const { box } = state.detections[photoIdx][faceIdx];

  const pad = box.width * 0.38;
  const sx = Math.max(0, box.x - pad);
  const sy = Math.max(0, box.y - pad * 1.5);  // more forehead room
  const sw = Math.min(img.naturalWidth - sx, box.width + pad * 2);
  const sh = Math.min(img.naturalHeight - sy, box.height + pad * 2.6);

  const c = document.createElement('canvas');
  c.width = FACE_CROP_SIZE;
  c.height = FACE_CROP_SIZE;
  c.getContext('2d').drawImage(img, sx, sy, sw, sh, 0, 0, FACE_CROP_SIZE, FACE_CROP_SIZE);
  return c;
}

// ════════════════════════════════════════════════════════
// Picker UI
// ════════════════════════════════════════════════════════

function renderPicker() {
  renderBaseSelector();
  renderPeopleList();
}

function renderBaseSelector() {
  els.baseSelector.innerHTML = '';
  state.photos.forEach((photo, idx) => {
    const wrap = document.createElement('div');
    wrap.className = 'base-thumb' + (idx === state.basePhotoIdx ? ' selected' : '');

    const img = document.createElement('img');
    img.src = photo.dataUrl;
    img.alt = `Photo ${idx + 1}`;

    const numLbl = document.createElement('span');
    numLbl.className = 'base-num';
    numLbl.textContent = `#${idx + 1}`;

    const check = document.createElement('span');
    check.className = 'base-check';
    check.textContent = '✓';

    wrap.appendChild(img);
    wrap.appendChild(numLbl);
    wrap.appendChild(check);

    const faceCount = state.detections[idx]?.length ?? 0;
    wrap.title = `${faceCount} face${faceCount !== 1 ? 's' : ''} detected`;

    wrap.addEventListener('click', () => {
      state.basePhotoIdx = idx;
      buildFaceSets();
      initFaceSetSelections();
      renderPicker();
    });

    els.baseSelector.appendChild(wrap);
  });
}

function renderPeopleList() {
  els.peopleList.innerHTML = '';

  if (state.faceSets.length === 0) {
    const msg = document.createElement('p');
    msg.style.cssText = 'color:var(--gray);font-size:.9rem;padding:12px 0;';
    msg.textContent = 'No faces detected in the base photo. Try a different base.';
    els.peopleList.appendChild(msg);
    return;
  }

  state.faceSets.forEach((faceSet, fi) => {
    const card = document.createElement('div');
    card.className = 'person-card';

    // Header row
    const meta = document.createElement('div');
    meta.className = 'person-meta';

    const name = document.createElement('p');
    name.className = 'person-name';
    name.textContent = `Guest ${fi + 1}`;

    const badge = document.createElement('span');
    badge.className = 'person-badge';
    const n = faceSet.options.length;
    badge.textContent = `${n} photo${n !== 1 ? 's' : ''}`;

    meta.appendChild(name);
    meta.appendChild(badge);
    card.appendChild(meta);

    // Face option row
    const row = document.createElement('div');
    row.className = 'face-row';

    const selIdx = state.faceSetSelections[faceSet.baseFaceIdx] ?? 0;

    faceSet.options.forEach(({ photoIdx, faceIdx }, optIdx) => {
      const isSelected = selIdx === optIdx;

      const opt = document.createElement('div');
      opt.className = 'face-option' + (isSelected ? ' selected' : '');

      const crop = makeFaceCropCanvas(photoIdx, faceIdx);
      opt.appendChild(crop);

      const lbl = document.createElement('span');
      lbl.className = 'face-photo-lbl';
      lbl.textContent = optIdx === 0 ? 'Base' : `#${photoIdx + 1}`;
      opt.appendChild(lbl);

      const chk = document.createElement('span');
      chk.className = 'face-check';
      chk.textContent = '✓';
      opt.appendChild(chk);

      opt.addEventListener('click', () => {
        state.faceSetSelections[faceSet.baseFaceIdx] = optIdx;
        renderPeopleList();
      });

      row.appendChild(opt);
    });

    card.appendChild(row);
    els.peopleList.appendChild(card);
  });
}

// ════════════════════════════════════════════════════════
// Compositing
// ════════════════════════════════════════════════════════

function getEyeCenters(landmarks) {
  const pts = landmarks.positions;
  const avg = idxs => ({
    x: idxs.reduce((s, i) => s + pts[i].x, 0) / idxs.length,
    y: idxs.reduce((s, i) => s + pts[i].y, 0) / idxs.length,
  });
  return {
    left:  avg([36, 37, 38, 39, 40, 41]),
    right: avg([42, 43, 44, 45, 46, 47]),
  };
}

// Returns a canvas 2D affine transform [a, b, c, d, e, f] (for setTransform) that
// maps srcL/srcR eye centers onto dstL/dstR eye centers (rotation + uniform scale + translation).
function computeAlignTransform(srcL, srcR, dstL, dstR) {
  const srcMx = (srcL.x + srcR.x) / 2, srcMy = (srcL.y + srcR.y) / 2;
  const dstMx = (dstL.x + dstR.x) / 2, dstMy = (dstL.y + dstR.y) / 2;

  // Vector from midpoint → right eye in each frame
  const vsx = srcR.x - srcMx, vsy = srcR.y - srcMy;
  const vdx = dstR.x - dstMx, vdy = dstR.y - dstMy;
  const s2 = vsx * vsx + vsy * vsy;

  // Complex-number division to get rotation+scale coefficients
  const ma = (vsx * vdx + vsy * vdy) / s2;
  const mb = (vsx * vdy - vsy * vdx) / s2;

  // Translation: srcMid must map to dstMid
  const e = dstMx - (ma * srcMx - mb * srcMy);
  const f = dstMy - (mb * srcMx + ma * srcMy);

  // Canvas setTransform(a,b,c,d,e,f): x'=ax+cy+e, y'=bx+dy+f
  return [ma, mb, -mb, ma, e, f];
}

// Blends srcFace (from srcImg) into dstFace position on dstCtx via affine alignment + feathered elliptical mask
function blendFace(dstCtx, srcImg, srcFace, dstFace, W, H) {
  const srcEyes = getEyeCenters(srcFace.landmarks);
  const dstEyes = getEyeCenters(dstFace.landmarks);

  const dstMx = (dstEyes.left.x + dstEyes.right.x) / 2;
  const dstMy = (dstEyes.left.y + dstEyes.right.y) / 2;
  const eyeDist = Math.hypot(
    dstEyes.right.x - dstEyes.left.x,
    dstEyes.right.y - dstEyes.left.y
  );

  const transform = computeAlignTransform(srcEyes.left, srcEyes.right, dstEyes.left, dstEyes.right);

  // Temp canvas: draw source image with alignment transform applied
  const tmp = document.createElement('canvas');
  tmp.width = W;
  tmp.height = H;
  const tCtx = tmp.getContext('2d');

  tCtx.setTransform(...transform);
  tCtx.drawImage(srcImg, 0, 0);
  tCtx.resetTransform();

  // Cut out an elliptical feathered face region using destination-in compositing
  const rx = eyeDist * 1.65;
  const ry = eyeDist * 2.45;

  tCtx.globalCompositeOperation = 'destination-in';
  tCtx.save();
  tCtx.translate(dstMx, dstMy);
  tCtx.scale(rx, ry);  // unit circle in this space = face ellipse in pixel space

  const grad = tCtx.createRadialGradient(0, 0, 0.38, 0, 0, 1.0);
  grad.addColorStop(0,    'rgba(0,0,0,1)');
  grad.addColorStop(0.62, 'rgba(0,0,0,0.92)');
  grad.addColorStop(1,    'rgba(0,0,0,0)');

  tCtx.fillStyle = grad;
  tCtx.beginPath();
  tCtx.arc(0, 0, 1, 0, Math.PI * 2);
  tCtx.fill();
  tCtx.restore();

  // Composite masked face onto destination (source-over by default)
  dstCtx.drawImage(tmp, 0, 0);
}

async function generateComposite() {
  const basePhoto = state.photos[state.basePhotoIdx];
  const W = basePhoto.img.naturalWidth;
  const H = basePhoto.img.naturalHeight;

  const canvas = els.resultCanvas;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  ctx.drawImage(basePhoto.img, 0, 0);

  for (const faceSet of state.faceSets) {
    const optIdx = state.faceSetSelections[faceSet.baseFaceIdx] ?? 0;
    if (optIdx === 0) continue; // base photo face already in place

    const opt    = faceSet.options[optIdx];
    const dstFace = state.detections[state.basePhotoIdx][faceSet.baseFaceIdx];
    const srcFace = state.detections[opt.photoIdx][opt.faceIdx];
    const srcImg  = state.photos[opt.photoIdx].img;

    blendFace(ctx, srcImg, srcFace, dstFace, W, H);
  }
}

// ════════════════════════════════════════════════════════
// Pipeline runners
// ════════════════════════════════════════════════════════

async function runAnalysis() {
  showScreen('loading');
  els.loadingSub.textContent = 'Models download once and are cached by your browser';

  try {
    await loadModels();

    els.loadingSub.textContent = '';
    await detectAllFaces();

    const totalFaces = state.detections.reduce((s, d) => s + d.length, 0);
    if (totalFaces === 0) {
      alert('No faces detected. Try uploading clearer, well-lit photos where faces are visible.');
      showScreen('upload');
      return;
    }

    setProgress(100, 'Building face sets…');
    await new Promise(r => setTimeout(r, 50));

    pickDefaultBase();
    buildFaceSets();
    initFaceSetSelections();
    renderPicker();
    showScreen('picker');
  } catch (err) {
    console.error(err);
    alert('Something went wrong during face detection. Please try again.');
    showScreen('upload');
  }
}

async function runComposite() {
  showScreen('loading');
  setProgress(20, 'Compositing faces…');
  els.loadingSub.textContent = '';
  await new Promise(r => setTimeout(r, 60));  // let loading screen paint

  try {
    await generateComposite();
    setProgress(100, 'Done!');
    await new Promise(r => setTimeout(r, 250));
    showScreen('result');
  } catch (err) {
    console.error(err);
    alert('Compositing failed. Please try again.');
    showScreen('picker');
  }
}

// ════════════════════════════════════════════════════════
// Event wiring
// ════════════════════════════════════════════════════════

function initEvents() {
  // File pick via input
  els.fileInput.addEventListener('change', e => {
    handleFiles(e.target.files);
    els.fileInput.value = '';  // reset so the same file can be re-added
  });

  // Drag-and-drop
  els.dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    els.dropzone.classList.add('drag-over');
  });
  els.dropzone.addEventListener('dragleave', () => {
    els.dropzone.classList.remove('drag-over');
  });
  els.dropzone.addEventListener('drop', e => {
    e.preventDefault();
    els.dropzone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
  });

  // Analyze
  els.btnAnalyze.addEventListener('click', () => {
    if (state.photos.length < 2) {
      alert('Please upload at least 2 group photos.');
      return;
    }
    // Reset previous analysis in case user went back and added photos
    state.detections = [];
    state.faceSets = [];
    state.faceSetSelections = {};
    runAnalysis();
  });

  // Back from picker → upload (keep photos, re-run detection on next click)
  els.btnBack.addEventListener('click', () => showScreen('upload'));

  // Generate composite
  els.btnCompose.addEventListener('click', runComposite);

  // Back from result → picker
  els.btnBackResult.addEventListener('click', () => showScreen('picker'));

  // Download
  els.btnDownload.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'postable.png';
    link.href = els.resultCanvas.toDataURL('image/png');
    link.click();
  });

  // Start over
  els.btnRestart.addEventListener('click', () => {
    state.photos = [];
    state.detections = [];
    state.faceSets = [];
    state.faceSetSelections = {};
    state.basePhotoIdx = 0;
    renderThumbnails();
    showScreen('upload');
  });
}

// ════════════════════════════════════════════════════════
// Boot
// ════════════════════════════════════════════════════════
initEvents();
