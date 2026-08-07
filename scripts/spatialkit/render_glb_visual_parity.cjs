#!/usr/bin/env node
"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");
const { pathToFileURL } = require("url");

const HELP = `Render candidate/reference GLBs in Chrome and write browser visual proof artifacts.

Usage:
  node scripts/spatialkit/render_glb_visual_parity.cjs \\
    --candidate "$MLX_SPATIAL_TEST_SCRATCH/outputs/candidate/model.glb" \\
    --reference inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/model.glb \\
    --output-dir "$MLX_SPATIAL_TEST_SCRATCH/artifacts/browser-render" \\
    [--visual-report "$MLX_SPATIAL_TEST_SCRATCH/artifacts/visual-parity.json"] \\
    [--artifact-manifest "$MLX_SPATIAL_TEST_SCRATCH/artifacts/artifact-manifest.json"]

Dev dependency setup, kept out of package runtime:
  npm install --prefix "$MLX_SPATIAL_TEST_SCRATCH/cache/render-deps" playwright@1.60.0 three@0.181.2
  NODE_PATH="$MLX_SPATIAL_TEST_SCRATCH/cache/render-deps/node_modules" node scripts/spatialkit/render_glb_visual_parity.cjs ...
`;

const VIEWS = [
  { name: "iso", direction: [1.45, 1.05, 1.45] },
  { name: "front", direction: [0.0, 0.15, 1.9] },
  { name: "top", direction: [0.0, 2.1, 0.05] },
];

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(HELP);
    return;
  }
  const candidate = requiredFile(args.candidate, "--candidate");
  const reference = requiredFile(args.reference, "--reference");
  const outputDir = requiredPath(args.outputDir, "--output-dir");
  const visualReport = args.visualReport ? requiredFile(args.visualReport, "--visual-report") : null;
  const artifactManifest = args.artifactManifest
    ? requiredFile(args.artifactManifest, "--artifact-manifest")
    : null;
  const width = positiveInt(args.width || "384", "--width");
  const height = positiveInt(args.height || "384", "--height");
  const channel = args.channel || "chrome";
  const deps = resolveDeps();

  fs.mkdirSync(outputDir, { recursive: true });
  const htmlPath = path.join(outputDir, "index.html");
  const reportPath = path.join(outputDir, "browser_render_report.json");
  const screenshotPath = path.join(outputDir, "comparison.png");
  fs.writeFileSync(htmlPath, reportHtmlShell(), "utf8");

  const server = await startServer({
    candidate,
    reference,
    threeRoot: deps.threeRoot,
    width,
    height,
  });
  try {
    const { chromium } = deps.playwright;
    const browser = await chromium.launch({
      channel,
      headless: true,
      args: ["--ignore-gpu-blocklist", "--enable-webgl"],
    });
    try {
      const page = await browser.newPage({
        viewport: {
          width: Math.max(900, width * 2 + 96),
          height: VIEWS.length * (height + 96) + 120,
        },
        deviceScaleFactor: 1,
      });
      await page.goto(server.url, { waitUntil: "networkidle" });
      await page.waitForFunction(() => globalThis.__renderDone === true, null, { timeout: 60_000 });
      const browserReport = await page.evaluate(() => globalThis.__renderReport);
      await page.screenshot({ path: screenshotPath, fullPage: true });
      const report = buildReport({
        browserReport,
        candidate,
        reference,
        outputDir,
        htmlPath,
        reportPath,
        screenshotPath,
        width,
        height,
        channel,
      });
      writeJson(reportPath, report);
      fs.writeFileSync(htmlPath, reportHtml(report), "utf8");
      if (visualReport !== null) {
        augmentVisualReport(visualReport, report);
      }
      if (artifactManifest !== null) {
        augmentArtifactManifest(artifactManifest, report);
      }
      if (!report.summary.all_passed) {
        process.stderr.write(`browser render checks failed; see ${reportPath}\n`);
        process.exitCode = 1;
      }
    } finally {
      await browser.close();
    }
  } finally {
    await new Promise((resolve) => server.server.close(resolve));
  }
}

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      result.help = true;
      continue;
    }
    if (!arg.startsWith("--")) {
      throw new Error(`unexpected positional argument: ${arg}`);
    }
    const key = arg.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`${arg} requires a value`);
    }
    result[key] = value;
    index += 1;
  }
  return result;
}

function requiredFile(value, name) {
  if (!value) {
    throw new Error(`${name} is required`);
  }
  const resolved = path.resolve(value);
  if (!fs.statSync(resolved, { throwIfNoEntry: false })?.isFile()) {
    throw new Error(`${name} must point to a file: ${resolved}`);
  }
  return resolved;
}

function requiredPath(value, name) {
  if (!value) {
    throw new Error(`${name} is required`);
  }
  return path.resolve(value);
}

function positiveInt(value, name) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function resolveDeps() {
  const extraPaths = (process.env.NODE_PATH || "")
    .split(path.delimiter)
    .filter(Boolean);
  const searchPaths = [process.cwd(), ...extraPaths];
  try {
    const playwrightPath = require.resolve("playwright", { paths: searchPaths });
    const threeMainPath = require.resolve("three", { paths: searchPaths });
    return {
      playwright: require(playwrightPath),
      threeRoot: path.resolve(path.dirname(threeMainPath), ".."),
    };
  } catch (error) {
    throw new Error(
      `${error.message}\n\n${HELP}`
    );
  }
}

function startServer({ candidate, reference, threeRoot, width, height }) {
  const server = http.createServer((request, response) => {
    const url = new URL(request.url || "/", "http://127.0.0.1");
    try {
      if (url.pathname === "/") {
        send(response, 200, "text/html", renderPageHtml({ width, height }));
        return;
      }
      if (url.pathname === "/viewer.js") {
        send(response, 200, "text/javascript", viewerSource({ width, height, views: VIEWS }));
        return;
      }
      if (url.pathname === "/candidate.glb") {
        sendFile(response, candidate, "model/gltf-binary");
        return;
      }
      if (url.pathname === "/reference.glb") {
        sendFile(response, reference, "model/gltf-binary");
        return;
      }
      if (url.pathname.startsWith("/three/")) {
        const relative = decodeURIComponent(url.pathname.slice("/three/".length));
        const resolved = path.resolve(threeRoot, relative);
        if (!resolved.startsWith(threeRoot + path.sep)) {
          throw new Error("bad three path");
        }
        sendFile(response, resolved, contentType(resolved));
        return;
      }
      send(response, 404, "text/plain", "not found");
    } catch (error) {
      send(response, 500, "text/plain", String(error.stack || error));
    }
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      resolve({ server, url: `http://127.0.0.1:${address.port}/` });
    });
  });
}

function sendFile(response, filePath, type) {
  if (!fs.statSync(filePath, { throwIfNoEntry: false })?.isFile()) {
    send(response, 404, "text/plain", "not found");
    return;
  }
  response.writeHead(200, {
    "Content-Type": type,
    "Access-Control-Allow-Origin": "*",
  });
  fs.createReadStream(filePath).pipe(response);
}

function send(response, status, type, body) {
  response.writeHead(status, {
    "Content-Type": `${type}; charset=utf-8`,
    "Access-Control-Allow-Origin": "*",
  });
  response.end(body);
}

function contentType(filePath) {
  if (filePath.endsWith(".js")) {
    return "text/javascript";
  }
  if (filePath.endsWith(".json")) {
    return "application/json";
  }
  return "application/octet-stream";
}

function renderPageHtml() {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>mlx-spatial SpatialKit browser render proof</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; color: #111; }
    h1 { font-size: 20px; margin: 0 0 16px; }
    .view { margin-bottom: 22px; }
    .row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    canvas { width: 100%; border: 1px solid #bbb; background: #f6f6f6; }
    figcaption { font-size: 13px; margin-top: 6px; }
  </style>
  <script type="importmap">
    {"imports":{"three":"/three/build/three.module.js","three/addons/":"/three/examples/jsm/"}}
  </script>
</head>
<body>
  <h1>mlx-spatial SpatialKit Browser Render Proof</h1>
  <div id="root"></div>
  <script type="module" src="/viewer.js"></script>
</body>
</html>`;
}

function viewerSource({ width, height, views }) {
  return `
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const WIDTH = ${JSON.stringify(width)};
const HEIGHT = ${JSON.stringify(height)};
const VIEWS = ${JSON.stringify(views)};
const loader = new GLTFLoader();
const root = document.getElementById('root');
const report = { views: [], errors: [] };

try {
  const [candidateGltf, referenceGltf] = await Promise.all([
    loader.loadAsync('/candidate.glb'),
    loader.loadAsync('/reference.glb'),
  ]);
  const candidateBounds = measureBounds(candidateGltf.scene);
  const referenceBounds = measureBounds(referenceGltf.scene);
  const framing = {
    center: referenceBounds.center,
    radius: Math.max(referenceBounds.size.x, referenceBounds.size.y, referenceBounds.size.z, 1e-5),
  };
  report.bounds = {
    candidate: candidateBounds,
    reference: referenceBounds,
    camera_frame_source: 'reference',
  };
  for (const view of VIEWS) {
    const section = document.createElement('section');
    section.className = 'view';
    section.innerHTML = '<h2>' + view.name + '</h2><div class="row"></div>';
    root.appendChild(section);
    const row = section.querySelector('.row');
    const candidate = renderModel(candidateGltf.scene, 'candidate', view, row, framing);
    const reference = renderModel(referenceGltf.scene, 'reference', view, row, framing);
    const comparison = compareRenderPixels(
      candidate.pixels,
      reference.pixels,
      candidate.normalPixels,
      reference.normalPixels,
      candidate.depthPixels,
      reference.depthPixels,
      candidate.cameraNear,
      candidate.cameraFar,
      framing.radius,
    );
    delete candidate.pixels;
    delete reference.pixels;
    delete candidate.normalPixels;
    delete reference.normalPixels;
    delete candidate.depthPixels;
    delete reference.depthPixels;
    report.views.push({
      name: view.name,
      candidate,
      reference,
      visible_pixel_ratio: reference.visible_pixel_count > 0 ? candidate.visible_pixel_count / reference.visible_pixel_count : null,
      ...comparison,
    });
  }
} catch (error) {
  report.errors.push(String(error && error.stack ? error.stack : error));
}

globalThis.__renderReport = report;
globalThis.__renderDone = true;

function measureBounds(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  return {
    min: { x: box.min.x, y: box.min.y, z: box.min.z },
    max: { x: box.max.x, y: box.max.y, z: box.max.z },
    size: { x: size.x, y: size.y, z: size.z },
    center: { x: center.x, y: center.y, z: center.z },
  };
}

function renderModel(object, label, view, row, framing) {
  const figure = document.createElement('figure');
  const caption = document.createElement('figcaption');
  caption.textContent = label + ' / ' + view.name;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(WIDTH, HEIGHT, false);
  renderer.setClearColor(0x000000, 0);
  figure.appendChild(renderer.domElement);
  figure.appendChild(caption);
  row.appendChild(figure);

  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 1.4));
  const directional = new THREE.DirectionalLight(0xffffff, 2.0);
  directional.position.set(3, 4, 5);
  scene.add(directional);
  scene.add(object);

  const box = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  box.getSize(size);
  const center = new THREE.Vector3(framing.center.x, framing.center.y, framing.center.z);
  const radius = framing.radius;
  const camera = new THREE.PerspectiveCamera(35, WIDTH / HEIGHT, radius / 100, radius * 100);
  const direction = new THREE.Vector3(view.direction[0], view.direction[1], view.direction[2]).normalize();
  camera.position.copy(center).add(direction.multiplyScalar(radius * 2.4));
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  renderer.render(scene, camera);

  const pixels = new Uint8Array(WIDTH * HEIGHT * 4);
  const gl = renderer.getContext();
  gl.readPixels(0, 0, WIDTH, HEIGHT, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  // Derive normals from rendered triangle geometry. Stored GLB NORMAL
  // attributes may use different smoothing conventions and are validated
  // separately by the structural GLB checks.
  const originalOutputColorSpace = renderer.outputColorSpace;
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  const normalMaterial = new THREE.MeshNormalMaterial({
    flatShading: true,
    side: THREE.DoubleSide,
  });
  scene.overrideMaterial = normalMaterial;
  renderer.render(scene, camera);
  const normalPixels = new Uint8Array(WIDTH * HEIGHT * 4);
  gl.readPixels(0, 0, WIDTH, HEIGHT, gl.RGBA, gl.UNSIGNED_BYTE, normalPixels);
  const depthMaterial = new THREE.MeshDepthMaterial({
    depthPacking: THREE.RGBDepthPacking,
    side: THREE.DoubleSide,
  });
  scene.overrideMaterial = depthMaterial;
  renderer.render(scene, camera);
  const depthPixels = new Uint8Array(WIDTH * HEIGHT * 4);
  gl.readPixels(0, 0, WIDTH, HEIGHT, gl.RGBA, gl.UNSIGNED_BYTE, depthPixels);
  scene.overrideMaterial = null;
  normalMaterial.dispose();
  depthMaterial.dispose();
  renderer.outputColorSpace = originalOutputColorSpace;
  renderer.render(scene, camera);
  let visible = 0;
  let rgb = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    if (pixels[index + 3] > 8) visible += 1;
    if (pixels[index] || pixels[index + 1] || pixels[index + 2]) rgb += 1;
  }
  return {
    pixels,
    normalPixels,
    depthPixels,
    cameraNear: camera.near,
    cameraFar: camera.far,
    visible_pixel_count: visible,
    rgb_nonzero_pixel_count: rgb,
    pixel_count: WIDTH * HEIGHT,
    visible_pixel_ratio: visible / (WIDTH * HEIGHT),
    bbox_size: { x: size.x, y: size.y, z: size.z },
  };
}

function compareRenderPixels(
  candidate,
  reference,
  candidateNormals,
  referenceNormals,
  candidateDepths,
  referenceDepths,
  cameraNear,
  cameraFar,
  radius,
) {
  let intersection = 0;
  let union = 0;
  let rgbAbsoluteError = 0;
  let normalAbsoluteError = 0;
  let depthAbsoluteError = 0;
  let rawDepthAbsoluteError = 0;
  let candidateViewDepthSum = 0;
  let referenceViewDepthSum = 0;
  for (let index = 0; index < candidate.length; index += 4) {
    const candidateVisible = candidate[index + 3] > 8;
    const referenceVisible = reference[index + 3] > 8;
    if (candidateVisible && referenceVisible) {
      intersection += 1;
      rgbAbsoluteError += Math.abs(candidate[index] - reference[index]);
      rgbAbsoluteError += Math.abs(candidate[index + 1] - reference[index + 1]);
      rgbAbsoluteError += Math.abs(candidate[index + 2] - reference[index + 2]);
      const directNormalError = (
        Math.abs(candidateNormals[index] - referenceNormals[index])
        + Math.abs(candidateNormals[index + 1] - referenceNormals[index + 1])
        + Math.abs(candidateNormals[index + 2] - referenceNormals[index + 2])
      );
      const flippedNormalError = (
        Math.abs(candidateNormals[index] - (255 - referenceNormals[index]))
        + Math.abs(candidateNormals[index + 1] - (255 - referenceNormals[index + 1]))
        + Math.abs(candidateNormals[index + 2] - (255 - referenceNormals[index + 2]))
      );
      normalAbsoluteError += Math.min(directNormalError, flippedNormalError);
      const candidateDepth = unpackRgbDepth(candidateDepths, index);
      const referenceDepth = unpackRgbDepth(referenceDepths, index);
      const candidateViewZ = perspectiveDepthToViewZ(candidateDepth, cameraNear, cameraFar);
      const referenceViewZ = perspectiveDepthToViewZ(referenceDepth, cameraNear, cameraFar);
      rawDepthAbsoluteError += Math.abs(candidateDepth - referenceDepth);
      depthAbsoluteError += Math.abs(candidateViewZ - referenceViewZ) / radius;
      candidateViewDepthSum += -candidateViewZ / radius;
      referenceViewDepthSum += -referenceViewZ / radius;
    }
    if (candidateVisible || referenceVisible) union += 1;
  }
  return {
    silhouette_intersection_pixels: intersection,
    silhouette_union_pixels: union,
    silhouette_iou: union > 0 ? intersection / union : null,
    shared_visible_rgb_mae: intersection > 0 ? rgbAbsoluteError / (intersection * 3 * 255) : null,
    shared_visible_orientation_invariant_normal_mae: (
      intersection > 0 ? normalAbsoluteError / (intersection * 3 * 255) : null
    ),
    shared_visible_depth_mae_radius_ratio: intersection > 0 ? depthAbsoluteError / intersection : null,
    shared_visible_raw_depth_mae: intersection > 0 ? rawDepthAbsoluteError / intersection : null,
    candidate_mean_view_depth_radius_ratio: intersection > 0 ? candidateViewDepthSum / intersection : null,
    reference_mean_view_depth_radius_ratio: intersection > 0 ? referenceViewDepthSum / intersection : null,
  };
}

function unpackRgbDepth(pixels, index) {
  const unpackDownscale = 255 / 256;
  return (
    (pixels[index] / 255) * unpackDownscale
    + (pixels[index + 1] / 255) * (unpackDownscale / 256)
    + (pixels[index + 2] / 255) / (256 * 256)
  );
}

function perspectiveDepthToViewZ(depth, near, far) {
  return (near * far) / ((far - near) * depth - far);
}
`;
}

function buildReport({ browserReport, candidate, reference, outputDir, htmlPath, reportPath, screenshotPath, width, height, channel }) {
  const checks = {};
  const views = Array.isArray(browserReport.views) ? browserReport.views : [];
  const minVisible = Math.max(128, Math.floor(width * height * 0.0025));
  const ratios = views.map((view) => view.visible_pixel_ratio).filter((value) => typeof value === "number" && Number.isFinite(value));
  const silhouetteIous = views.map((view) => view.silhouette_iou).filter(isFiniteNumber);
  const rgbErrors = views.map((view) => view.shared_visible_rgb_mae).filter(isFiniteNumber);
  const normalErrors = views
    .map((view) => view.shared_visible_orientation_invariant_normal_mae)
    .filter(isFiniteNumber);
  const depthErrors = views
    .map((view) => view.shared_visible_depth_mae_radius_ratio)
    .filter(isFiniteNumber);
  const rawDepthErrors = views.map((view) => view.shared_visible_raw_depth_mae).filter(isFiniteNumber);
  const candidateMeanDepths = views
    .map((view) => view.candidate_mean_view_depth_radius_ratio)
    .filter(isFiniteNumber);
  const referenceMeanDepths = views
    .map((view) => view.reference_mean_view_depth_radius_ratio)
    .filter(isFiniteNumber);
  const boundsComparison = compareBounds(browserReport.bounds);
  checks.browser_render_completed = {
    passed: Array.isArray(browserReport.errors) && browserReport.errors.length === 0,
    errors: browserReport.errors || [],
  };
  checks.all_views_rendered = {
    passed: views.length === VIEWS.length,
    actual: views.length,
    required: VIEWS.length,
  };
  checks.candidate_visible_all_views = {
    passed: views.every((view) => view.candidate.visible_pixel_count >= minVisible),
    required_min_visible_pixels: minVisible,
  };
  checks.reference_visible_all_views = {
    passed: views.every((view) => view.reference.visible_pixel_count >= minVisible),
    required_min_visible_pixels: minVisible,
  };
  checks.visible_pixel_ratio = {
    passed: ratios.length === VIEWS.length && ratios.every((ratio) => ratio >= 0.95 && ratio <= 1.05),
    actual: ratios,
    required_min: 0.95,
    required_max: 1.05,
  };
  checks.shared_camera_frame = {
    passed: browserReport.bounds?.camera_frame_source === "reference",
    actual: browserReport.bounds?.camera_frame_source || null,
    required: "reference",
  };
  checks.bounding_box_center_offset = {
    passed: boundsComparison.center_offset_ratio !== null && boundsComparison.center_offset_ratio <= 0.02,
    actual: boundsComparison.center_offset_ratio,
    required_max: 0.02,
  };
  checks.bounding_box_extent_ratio = {
    passed: boundsComparison.extent_ratios.length === 3
      && boundsComparison.extent_ratios.every((ratio) => ratio >= 0.95 && ratio <= 1.05),
    actual: boundsComparison.extent_ratios,
    required_min: 0.95,
    required_max: 1.05,
  };
  checks.silhouette_iou = {
    passed: silhouetteIous.length === VIEWS.length && silhouetteIous.every((value) => value >= 0.95),
    actual: silhouetteIous,
    required_min: 0.95,
  };
  checks.shared_visible_rgb_mae = {
    passed: rgbErrors.length === VIEWS.length && rgbErrors.every((value) => value <= 0.15),
    actual: rgbErrors,
    required_max: 0.15,
  };
  checks.shared_visible_orientation_invariant_normal_mae = {
    passed: normalErrors.length === VIEWS.length && normalErrors.every((value) => value <= 0.12),
    actual: normalErrors,
    required_max: 0.12,
  };
  checks.shared_visible_depth_mae_radius_ratio = {
    passed: depthErrors.length === VIEWS.length && depthErrors.every((value) => value >= 0.0),
    actual: depthErrors,
    required: "reported informationally; the non-watertight reference GLB is not a surface-distance oracle",
    quality_gate: false,
  };
  const allPassed = Object.values(checks).every((check) => Boolean(check.passed));
  return {
    schema: "mlx-spatial-spatialkit-browser-render-v2",
    candidate,
    reference,
    settings: {
      width,
      height,
      channel,
      views: VIEWS,
      camera_frame_source: "reference",
    },
    summary: {
      all_passed: allPassed,
      spatial_proof_ready: allPassed,
      rendered_visual_ready: allPassed,
      comparison_kind: "shared-camera-browser-render",
      rendered_view_count: views.length,
      min_visible_pixel_count: minVisible,
      visible_pixel_ratios: ratios,
      silhouette_ious: silhouetteIous,
      shared_visible_rgb_mae: rgbErrors,
      shared_visible_orientation_invariant_normal_mae: normalErrors,
      shared_visible_depth_mae_radius_ratio: depthErrors,
      shared_visible_raw_depth_mae: rawDepthErrors,
      candidate_mean_view_depth_radius_ratios: candidateMeanDepths,
      reference_mean_view_depth_radius_ratios: referenceMeanDepths,
      bounding_box_center_offset_ratio: boundsComparison.center_offset_ratio,
      bounding_box_extent_ratios: boundsComparison.extent_ratios,
    },
    checks,
    bounds: browserReport.bounds || null,
    views,
    artifacts: {
      report_json: reportPath,
      comparison_png: screenshotPath,
      html_report: htmlPath,
      output_dir: outputDir,
    },
  };
}

function compareBounds(bounds) {
  const candidate = bounds?.candidate;
  const reference = bounds?.reference;
  if (!candidate || !reference) {
    return { center_offset_ratio: null, extent_ratios: [] };
  }
  const referenceDiagonal = Math.hypot(reference.size.x, reference.size.y, reference.size.z);
  const centerOffset = Math.hypot(
    candidate.center.x - reference.center.x,
    candidate.center.y - reference.center.y,
    candidate.center.z - reference.center.z,
  );
  const extentRatios = ["x", "y", "z"].map((axis) => {
    const denominator = reference.size[axis];
    return denominator > 0 ? candidate.size[axis] / denominator : null;
  }).filter(isFiniteNumber);
  return {
    center_offset_ratio: referenceDiagonal > 0 ? centerOffset / referenceDiagonal : null,
    extent_ratios: extentRatios,
  };
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function augmentVisualReport(visualReportPath, browserReport) {
  const payload = JSON.parse(fs.readFileSync(visualReportPath, "utf8"));
  payload.browser_render = {
    summary: browserReport.summary,
    checks: browserReport.checks,
    views: browserReport.views,
    artifacts: browserReport.artifacts,
  };
  payload.artifacts = {
    ...(payload.artifacts || {}),
    browser_render_report_json: browserReport.artifacts.report_json,
    browser_render_comparison_png: browserReport.artifacts.comparison_png,
    browser_render_html: browserReport.artifacts.html_report,
  };
  payload.checks = {
    ...(payload.checks || {}),
    browser_rendered_visual_proof: {
      passed: browserReport.summary.all_passed,
      required: true,
      report_json: browserReport.artifacts.report_json,
    },
  };
  payload.summary = {
    ...(payload.summary || {}),
    browser_rendered_visual_proof: browserReport.summary.all_passed,
    spatial_proof_ready: browserReport.summary.spatial_proof_ready,
  };
  payload.comparison_scope = {
    kind: "shared-camera-browser-render",
    spatial_proof: browserReport.summary.spatial_proof_ready,
    viewer_render_proof: browserReport.summary.rendered_visual_ready,
    texture_registration_proof: browserReport.summary.rendered_visual_ready,
    reason: browserReport.summary.all_passed
      ? "Candidate and reference passed shared-camera bounds, silhouette, and rendered-color checks."
      : "Shared-camera browser render checks did not all pass.",
  };
  if (browserReport.summary.all_passed && Array.isArray(payload.deferred_parity_boundaries)) {
    payload.deferred_parity_boundaries = payload.deferred_parity_boundaries.filter(
      (item) => item !== "not_spatial_rendered_visual_proof"
    );
  } else if (
    Array.isArray(payload.deferred_parity_boundaries)
    && !payload.deferred_parity_boundaries.includes("not_spatial_rendered_visual_proof")
  ) {
    payload.deferred_parity_boundaries.push("not_spatial_rendered_visual_proof");
  }
  writeJson(visualReportPath, payload);
}

function augmentArtifactManifest(manifestPath, browserReport) {
  const payload = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  if (!payload.roles || !payload.roles.B) {
    throw new Error(`artifact manifest missing roles.B: ${manifestPath}`);
  }
  payload.roles.B.browser_render_report_path = browserReport.artifacts.report_json;
  payload.roles.B.browser_render_comparison_png_path = browserReport.artifacts.comparison_png;
  payload.roles.B.browser_render_html_path = browserReport.artifacts.html_report;
  payload.browser_render = {
    summary: browserReport.summary,
    checks: browserReport.checks,
    artifacts: browserReport.artifacts,
  };
  writeJson(manifestPath, payload);
}

function reportHtmlShell() {
  return "<!doctype html><title>mlx-spatial SpatialKit browser render report</title><p>Report is being generated.</p>";
}

function reportHtml(report) {
  const status = report.summary.all_passed ? "PASS" : "FAIL";
  const rows = report.views.map((view) => {
    return `<tr><td>${escapeHtml(view.name)}</td><td>${view.candidate.visible_pixel_count}</td><td>${view.reference.visible_pixel_count}</td><td>${formatMetric(view.visible_pixel_ratio)}</td><td>${formatMetric(view.silhouette_iou)}</td><td>${formatMetric(view.shared_visible_depth_mae_radius_ratio)}</td><td>${formatMetric(view.shared_visible_rgb_mae)}</td><td>${formatMetric(view.shared_visible_orientation_invariant_normal_mae)}</td></tr>`;
  }).join("\n");
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>mlx-spatial SpatialKit browser render report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111; }
    main { max-width: 1100px; margin: 0 auto; }
    img { width: 100%; border: 1px solid #bbb; }
    table { border-collapse: collapse; margin-top: 16px; width: 100%; }
    th, td { border-bottom: 1px solid #ddd; padding: 8px; text-align: left; }
    code { background: #f1f1f1; padding: 2px 4px; border-radius: 3px; }
  </style>
</head>
<body>
<main>
  <h1>mlx-spatial SpatialKit Browser Render Proof: ${status}</h1>
  <p>Candidate: <code>${escapeHtml(report.candidate)}</code></p>
  <p>Reference: <code>${escapeHtml(report.reference)}</code></p>
  <img src="${escapeHtml(path.basename(report.artifacts.comparison_png))}" alt="candidate and reference browser render comparison">
  <table>
    <thead><tr><th>view</th><th>candidate visible px</th><th>reference visible px</th><th>visible ratio</th><th>silhouette IoU</th><th>depth MAE / radius</th><th>RGB MAE</th><th>orientation-invariant normal MAE</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <p>Machine report: <a href="${escapeHtml(path.basename(report.artifacts.report_json))}">browser_render_report.json</a></p>
</main>
</body>
</html>
`;
}

function formatMetric(value) {
  return Number.isFinite(value) ? Number(value).toFixed(4) : "n/a";
}

function writeJson(filePath, payload) {
  fs.writeFileSync(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
