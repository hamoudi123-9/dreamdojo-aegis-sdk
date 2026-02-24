// robot.js — 7-DOF Robot Arm with Three.js primitives (KUKA-style)
import * as THREE from 'three';

const COLORS = {
  orange: 0xff6600,       // KUKA orange — joints
  darkGray: 0x505050,     // Body / links (brighter)
  medGray: 0x707070,      // Accent panels (brighter)
  gripperGray: 0x6a6a6a,  // Gripper (brighter)
  violation: 0xff4444,
  ghost: 0x805ad5,
};

/** Create a single robot arm group */
export function createRobot(scene, options = {}) {
  const { ghost = false } = options;
  const root = new THREE.Group();

  // ── Helper: create a MeshStandardMaterial with ghost support ──
  function mat(color, metalness = 0.6, roughness = 0.3) {
    return new THREE.MeshStandardMaterial({
      color,
      metalness,
      roughness,
      transparent: ghost,
      opacity: ghost ? 0.25 : 1,
      wireframe: ghost,
    });
  }

  // Named material instances
  const matBase    = mat(COLORS.darkGray, 0.7, 0.2);
  const matJoint   = mat(COLORS.orange, 0.5, 0.25);
  const matLink    = mat(COLORS.darkGray, 0.6, 0.25);
  const matGripper = mat(COLORS.gripperGray, 0.5, 0.35);
  const matAccent  = mat(COLORS.medGray, 0.5, 0.3);
  const matOrangeAccent = mat(COLORS.orange, 0.5, 0.3);

  const matViolation = new THREE.MeshStandardMaterial({
    color: COLORS.violation,
    metalness: 0.2,
    roughness: 0.3,
    emissive: 0xff2222,
    emissiveIntensity: 0.5,
  });

  // ══════════════════════════════════════════════════════════════
  // ── BASE: Industrial pedestal ──
  // ══════════════════════════════════════════════════════════════

  const baseGroup = new THREE.Group();
  baseGroup.position.y = 0;
  root.add(baseGroup);

  // Floor plate (square)
  const floorPlate = new THREE.Mesh(
    new THREE.BoxGeometry(0.5, 0.04, 0.5),
    matBase.clone(),
  );
  floorPlate.position.y = 0.02;
  baseGroup.add(floorPlate);

  // Hexagonal pedestal column
  const pedestal = new THREE.Mesh(
    new THREE.CylinderGeometry(0.14, 0.17, 0.14, 6),
    matBase.clone(),
  );
  pedestal.position.y = 0.04 + 0.07;
  baseGroup.add(pedestal);

  // Top flange ring
  const flange = new THREE.Mesh(
    new THREE.CylinderGeometry(0.16, 0.16, 0.02, 24),
    matAccent.clone(),
  );
  flange.position.y = 0.18 + 0.01;
  baseGroup.add(flange);

  // Orange accent ring
  const accentRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.155, 0.012, 8, 24),
    matOrangeAccent.clone(),
  );
  accentRing.rotation.x = Math.PI / 2;
  accentRing.position.y = 0.20;
  baseGroup.add(accentRing);

  // baseMesh reference (for compatibility — use the pedestal)
  const baseMesh = pedestal;

  // ══════════════════════════════════════════════════════════════
  // ── JOINT 0: Base rotator ──
  // ══════════════════════════════════════════════════════════════

  const joint0 = new THREE.Group();
  joint0.position.set(0, 0.22, 0);
  root.add(joint0);

  // Orange housing box (main)
  const j0Box = new THREE.Mesh(
    new THREE.BoxGeometry(0.16, 0.10, 0.14),
    matJoint.clone(),
  );
  joint0.add(j0Box);

  // Dark side panels
  const j0PanelL = new THREE.Mesh(
    new THREE.BoxGeometry(0.005, 0.08, 0.12),
    matAccent.clone(),
  );
  j0PanelL.position.x = -0.083;
  joint0.add(j0PanelL);

  const j0PanelR = new THREE.Mesh(
    new THREE.BoxGeometry(0.005, 0.08, 0.12),
    matAccent.clone(),
  );
  j0PanelR.position.x = 0.083;
  joint0.add(j0PanelR);

  // ══════════════════════════════════════════════════════════════
  // ── LINK 0: Upper arm segment ──
  // ══════════════════════════════════════════════════════════════

  const link0 = new THREE.Mesh(
    new THREE.BoxGeometry(0.14, 0.40, 0.10),
    matLink.clone(),
  );
  link0.position.y = 0.22;
  joint0.add(link0);

  // Front panel accent
  const l0Panel = new THREE.Mesh(
    new THREE.BoxGeometry(0.10, 0.36, 0.005),
    matAccent.clone(),
  );
  l0Panel.position.set(0, 0.22, 0.053);
  joint0.add(l0Panel);

  // Top/bottom ridges
  const l0RidgeTop = new THREE.Mesh(
    new THREE.BoxGeometry(0.15, 0.015, 0.11),
    matAccent.clone(),
  );
  l0RidgeTop.position.set(0, 0.42, 0);
  joint0.add(l0RidgeTop);

  const l0RidgeBot = new THREE.Mesh(
    new THREE.BoxGeometry(0.15, 0.015, 0.11),
    matAccent.clone(),
  );
  l0RidgeBot.position.set(0, 0.02, 0);
  joint0.add(l0RidgeBot);

  // ══════════════════════════════════════════════════════════════
  // ── JOINT 1: Shoulder ──
  // ══════════════════════════════════════════════════════════════

  const joint1 = new THREE.Group();
  joint1.position.set(0, 0.44, 0);
  joint0.add(joint1);

  // Slightly smaller orange housing
  const j1Box = new THREE.Mesh(
    new THREE.BoxGeometry(0.14, 0.09, 0.12),
    matJoint.clone(),
  );
  joint1.add(j1Box);

  // Front/back caps
  const j1CapF = new THREE.Mesh(
    new THREE.BoxGeometry(0.12, 0.07, 0.005),
    matAccent.clone(),
  );
  j1CapF.position.z = 0.063;
  joint1.add(j1CapF);

  const j1CapB = new THREE.Mesh(
    new THREE.BoxGeometry(0.12, 0.07, 0.005),
    matAccent.clone(),
  );
  j1CapB.position.z = -0.063;
  joint1.add(j1CapB);

  // ══════════════════════════════════════════════════════════════
  // ── LINK 1: Forearm segment ──
  // ══════════════════════════════════════════════════════════════

  const link1 = new THREE.Mesh(
    new THREE.BoxGeometry(0.12, 0.35, 0.09),
    matLink.clone(),
  );
  link1.position.y = 0.175;
  joint1.add(link1);

  // Side panel strip
  const l1Strip = new THREE.Mesh(
    new THREE.BoxGeometry(0.005, 0.30, 0.07),
    matAccent.clone(),
  );
  l1Strip.position.set(0.063, 0.175, 0);
  joint1.add(l1Strip);

  // ══════════════════════════════════════════════════════════════
  // ── JOINT 2: Elbow ──
  // ══════════════════════════════════════════════════════════════

  const joint2 = new THREE.Group();
  joint2.position.set(0, 0.35, 0);
  joint1.add(joint2);

  // Smaller orange housing
  const j2Box = new THREE.Mesh(
    new THREE.BoxGeometry(0.11, 0.08, 0.10),
    matJoint.clone(),
  );
  joint2.add(j2Box);

  // Flange ring
  const j2Flange = new THREE.Mesh(
    new THREE.CylinderGeometry(0.065, 0.065, 0.015, 16),
    matAccent.clone(),
  );
  j2Flange.position.y = 0.04;
  joint2.add(j2Flange);

  // ══════════════════════════════════════════════════════════════
  // ── LINK 2: Wrist segment ──
  // ══════════════════════════════════════════════════════════════

  const link2 = new THREE.Mesh(
    new THREE.BoxGeometry(0.09, 0.20, 0.07),
    matLink.clone(),
  );
  link2.position.y = 0.125;
  joint2.add(link2);

  // Wrist cylinder at end
  const wristCyl = new THREE.Mesh(
    new THREE.CylinderGeometry(0.04, 0.04, 0.03, 16),
    matAccent.clone(),
  );
  wristCyl.position.y = 0.23;
  joint2.add(wristCyl);

  // ══════════════════════════════════════════════════════════════
  // ── GRIPPER: Industrial gripper ──
  // ══════════════════════════════════════════════════════════════

  const gripperMount = new THREE.Group();
  gripperMount.position.set(0, 0.25, 0);
  joint2.add(gripperMount);

  // Gripper base plate
  const gripperBase = new THREE.Mesh(
    new THREE.BoxGeometry(0.10, 0.02, 0.06),
    matGripper.clone(),
  );
  gripperBase.position.y = 0.01;
  gripperMount.add(gripperBase);

  // Left finger (dim 5) — thicker industrial style
  const fingerL = new THREE.Mesh(
    new THREE.BoxGeometry(0.025, 0.10, 0.05),
    matGripper.clone(),
  );
  fingerL.position.set(-0.04, 0.07, 0);
  gripperMount.add(fingerL);

  // Left finger orange tip
  const tipL = new THREE.Mesh(
    new THREE.BoxGeometry(0.025, 0.02, 0.05),
    matOrangeAccent.clone(),
  );
  tipL.position.y = 0.06;
  fingerL.add(tipL);

  // Right finger (dim 6)
  const fingerR = new THREE.Mesh(
    new THREE.BoxGeometry(0.025, 0.10, 0.05),
    matGripper.clone(),
  );
  fingerR.position.set(0.04, 0.07, 0);
  gripperMount.add(fingerR);

  // Right finger orange tip
  const tipR = new THREE.Mesh(
    new THREE.BoxGeometry(0.025, 0.02, 0.05),
    matOrangeAccent.clone(),
  );
  tipR.position.y = 0.06;
  fingerR.add(tipR);

  scene.add(root);

  // Pre-create cached materials for per-frame reuse (avoids memory leak)
  const _cachedJointMats = [matJoint.clone(), matJoint.clone(), matJoint.clone()];
  const _cachedGripperMats = [matGripper.clone(), matGripper.clone()];
  const _cachedViolationMats = [
    matViolation.clone(), matViolation.clone(), matViolation.clone(),  // joints
    matViolation.clone(), matViolation.clone(),                        // fingers
  ];

  // Robot state object — API-compatible return
  return {
    root,
    joints: [joint0, joint1, joint2],
    links: [link0, link1, link2],
    fingers: [fingerL, fingerR],
    gripperMount,
    baseMesh,
    jointSpheres: [j0Box, j1Box, j2Box],   // housing boxes replace spheres
    materials: { matBase, matJoint, matLink, matGripper, matViolation },
    nanParticles: null,
    _originalMaterials: new Map(),
    _cachedJointMats,
    _cachedGripperMats,
    _cachedViolationMats,
  };
}

/** Apply a 7-DOF action to the robot model */
export function applyAction(robot, action, violations = []) {
  const values = action;

  // Check for NaN — if any NaN, make robot transparent + show particles
  const hasNaN = Array.from(values).some(v => !Number.isFinite(v));
  if (hasNaN) {
    robot.root.traverse(child => {
      if (child.isMesh) {
        child.material.transparent = true;
        child.material.opacity = 0.1;
      }
    });
    showNanParticles(robot);
    return;
  }

  // Restore opacity
  robot.root.traverse(child => {
    if (child.isMesh && !child.material.wireframe) {
      child.material.transparent = false;
      child.material.opacity = 1.0;
    }
  });
  hideNanParticles(robot);

  // dim 0,1,2 → Base: translate X, translate Z, rotate Y
  robot.joints[0].position.x = values[0] * 0.15;
  robot.joints[0].position.z = values[1] * 0.15;
  robot.joints[0].rotation.y = values[2];

  // dim 3 → Shoulder rotation X
  robot.joints[1].rotation.x = values[3];

  // dim 4 → Elbow rotation X
  robot.joints[2].rotation.x = values[4];

  // dim 5,6 → Gripper fingers (open/close as X offset)
  const gripOpen5 = Number.isFinite(values[5]) ? values[5] : 0.5;
  const gripOpen6 = Number.isFinite(values[6]) ? values[6] : 0.5;
  robot.fingers[0].position.x = -0.02 - gripOpen5 * 0.06;
  robot.fingers[1].position.x = 0.02 + gripOpen6 * 0.06;

  // Highlight violated joints (reuse cached materials to avoid memory leak)
  resetJointColors(robot);
  const violatedDims = new Set();
  for (const v of violations) {
    if (v.dimension !== undefined) violatedDims.add(v.dimension);
  }

  // Map dimensions to visual joints
  if (violatedDims.has(0) || violatedDims.has(1) || violatedDims.has(2)) {
    robot.jointSpheres[0].material = robot._cachedViolationMats[0];
  }
  if (violatedDims.has(3)) {
    robot.jointSpheres[1].material = robot._cachedViolationMats[1];
  }
  if (violatedDims.has(4)) {
    robot.jointSpheres[2].material = robot._cachedViolationMats[2];
  }
  if (violatedDims.has(5)) {
    robot.fingers[0].material = robot._cachedViolationMats[3];
  }
  if (violatedDims.has(6)) {
    robot.fingers[1].material = robot._cachedViolationMats[4];
  }
}

function resetJointColors(robot) {
  for (let i = 0; i < robot.jointSpheres.length; i++) {
    robot.jointSpheres[i].material = robot._cachedJointMats[i];
  }
  robot.fingers[0].material = robot._cachedGripperMats[0];
  robot.fingers[1].material = robot._cachedGripperMats[1];
}

// ── NaN Particles ──
function showNanParticles(robot) {
  if (robot.nanParticles) return;

  const count = 40;
  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 0.6;
    positions[i * 3 + 1] = Math.random() * 1.2;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 0.6;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const mat = new THREE.PointsMaterial({
    color: 0xff4444,
    size: 0.03,
    transparent: true,
    opacity: 0.8,
  });

  robot.nanParticles = new THREE.Points(geo, mat);
  robot.root.add(robot.nanParticles);
}

function hideNanParticles(robot) {
  if (robot.nanParticles) {
    robot.root.remove(robot.nanParticles);
    robot.nanParticles.geometry.dispose();
    robot.nanParticles.material.dispose();
    robot.nanParticles = null;
  }
}

/** Animate NaN particles (call each frame) */
export function updateNanParticles(robot, time) {
  if (!robot.nanParticles) return;
  const pos = robot.nanParticles.geometry.attributes.position.array;
  for (let i = 0; i < pos.length; i += 3) {
    pos[i] += (Math.random() - 0.5) * 0.01;
    pos[i + 1] += 0.005;
    pos[i + 2] += (Math.random() - 0.5) * 0.01;
    if (pos[i + 1] > 1.5) pos[i + 1] = 0;
  }
  robot.nanParticles.geometry.attributes.position.needsUpdate = true;
}

/** Create a scene with lights and grid */
export function createScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  // Ambient — stronger for overall brightness
  scene.add(new THREE.AmbientLight(0x8888aa, 1.2));

  // Main directional light — brighter
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.6);
  dirLight.position.set(3, 5, 3);
  dirLight.castShadow = true;
  scene.add(dirLight);

  // Fill light from opposite side
  const fillLight = new THREE.DirectionalLight(0xaabbff, 0.6);
  fillLight.position.set(-3, 3, -2);
  scene.add(fillLight);

  // Point light for warm fill
  const pointLight = new THREE.PointLight(0x6366f1, 0.6, 10);
  pointLight.position.set(-2, 3, 1);
  scene.add(pointLight);

  // Rim light from behind for edge definition
  const rimLight = new THREE.DirectionalLight(0xffffff, 0.4);
  rimLight.position.set(0, 2, -4);
  scene.add(rimLight);

  // Grid helper
  const grid = new THREE.GridHelper(4, 20, 0x2a2a4a, 0x2a2a4a);
  scene.add(grid);

  return scene;
}

/** Create a standard camera */
export function createCamera(aspect) {
  const camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
  camera.position.set(1.5, 1.5, 2.0);
  camera.lookAt(0, 0.5, 0);
  return camera;
}
