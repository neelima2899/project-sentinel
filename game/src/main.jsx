import { buildObservation } from './ppoObservation.js';
import { predictAdversaryAction } from './ppoClient.js';
import { updatePowerStrike } from './powerStrike.js';
import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars, Text, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import './style.css';

const MAX = { player: 100, guardian: 120, enemy: 140 };
const START = { ...MAX, message: 'Move with WASD. Press SPACE to attack.', guardianDecision: 'GUARD', enemyDecision: 'ASSESS', playerHits: 0, enemyHits: 0 };
const distance = (a, b) => Math.hypot(a.x - b.x, a.z - b.z);
const moveToward = (a, b, speed, dt) => {
  const dx = b.x - a.x, dz = b.z - a.z, d = Math.hypot(dx, dz);
  if (d > .01) { const step = Math.min(d, speed * dt); a.x += dx / d * step; a.z += dz / d * step; }
};
const clampArena = p => { p.x = THREE.MathUtils.clamp(p.x, -8, 8); p.z = THREE.MathUtils.clamp(p.z, -8, 8); };

function Character({ color, label }) {
  return <group>
    <mesh position={[0, .9, 0]} castShadow><capsuleGeometry args={[.38, 1.05, 8, 16]}/><meshStandardMaterial color={color} metalness={.5} roughness={.3}/></mesh>
    <mesh position={[0, 1.72, 0]}><sphereGeometry args={[.32, 24, 24]}/><meshStandardMaterial color={color} emissive={color} emissiveIntensity={.25}/></mesh>
    <mesh position={[0, 1.75, .29]}><boxGeometry args={[.32, .08, .06]}/><meshBasicMaterial color="white"/></mesh>
    <Text position={[0, 2.3, 0]} fontSize={.25} color="white" anchorX="center">{label}</Text>
  </group>;
}
function Arena({ keys, resetId, onStats }) {
  const playerMesh = useRef(), guardianMesh = useRef(), enemyMesh = useRef();
  const sim = useRef(null), lastPublish = useRef(0);
  useEffect(() => {
    sim.current = {
      p: new THREE.Vector3(0, 0, 3), g: new THREE.Vector3(-4, 0, -3), e: new THREE.Vector3(4, 0, -3),
      ...START, playerAttack: 0, enemyAttack: 0, guardianAttack: 0,
      previousSpace: false, elapsed: 0, enemyTarget: 'guardian', lastTargetChange: 0,
      terminal: false,
      aiSteps: 0,
      nextAttackStep: 1,
      powerStrikeActive: false,
      powerStrikeCountdown: 0,
      powerStrikeCooldown: 0,
      aiDecisionElapsed: 0,
      aiDecisionInterval: 0.5,
      ppoAction: 0,
      ppoActionReady: false,
      damageStats: {
        player: 0,
        guardian: 0,
        adversary: 0,
        powerStrikes: 0,
        ppoDecisions: 0
      },
      ppoPending: false,
      ppoRequestId: 0,
      ppoStatus: 'CONNECTING',
      ppoNextRetry: 0,
    };
    lastPublish.current = 0;
    onStats({ ...START });
  }, [resetId, onStats]);

  const requestPPO = (s) => {
    if (s.ppoPending || s.terminal || s.elapsed < s.ppoNextRetry) return;

    s.ppoPending = true;
    const requestId = ++s.ppoRequestId;
    const observation = buildObservation(s);

    predictAdversaryAction(observation)
      .then(result => {
        if (sim.current !== s || s.terminal || requestId !== s.ppoRequestId) return;
        s.ppoAction = result.action;
        s.ppoActionReady = true;
        s.ppoStatus = 'PPO v4 ONLINE';
      })
      .catch(error => {
        if (sim.current !== s || requestId !== s.ppoRequestId) return;
        s.ppoStatus = 'API OFFLINE';
        s.ppoNextRetry = s.elapsed + 2;
        console.warn('PPO inference failed:', error.message);
      })
      .finally(() => {
        if (sim.current === s && requestId === s.ppoRequestId) {
          s.ppoPending = false;
        }
      });
  };

  useFrame((_, delta) => {
    const s = sim.current;
    if (!s) return;
    const dt = Math.min(delta, .05);
    if (!s.terminal) {
      s.elapsed += dt;

      // Advance Guardian Power Strike once per AI decision step.
      s.aiDecisionElapsed += dt;
      while (s.aiDecisionElapsed >= s.aiDecisionInterval) {
        s.aiDecisionElapsed -= s.aiDecisionInterval;
        s.aiSteps += 1;
        updatePowerStrike(s);
        requestPPO(s);
      }
      const dx = Number(keys.current.has('d') || keys.current.has('arrowright')) - Number(keys.current.has('a') || keys.current.has('arrowleft'));
      const dz = Number(keys.current.has('s') || keys.current.has('arrowdown')) - Number(keys.current.has('w') || keys.current.has('arrowup'));
      const moving = new THREE.Vector3(dx, 0, dz);
      if (s.player > 0 && moving.lengthSq()) { moving.normalize().multiplyScalar(4.5 * dt); s.p.add(moving); clampArena(s.p); }
      const space = keys.current.has(' ');
      if (space && !s.previousSpace && s.player > 0 && s.enemy > 0 && s.elapsed >= s.playerAttack) {
        s.playerAttack = s.elapsed + .65;
        if (distance(s.p, s.e) < 2.1) {
          const dealt = Math.min(18, s.enemy);
          s.enemy = Math.max(0, s.enemy - 18);
          s.damageStats.player += dealt;
          s.playerHits += 1;
          s.message = 'Player hit Adversary for 18 damage';
          s.enemyTarget = 'player'; // Attacking the enemy draws its attention.
          s.lastTargetChange = s.elapsed;
        } else s.message = 'Attack missed — get closer to the Adversary';
      }
      s.previousSpace = space;

      // Execute each fresh PPO prediction exactly once.
      if (s.enemy > 0 && s.player > 0) {
        if (s.ppoStatus === 'CONNECTING' || s.ppoStatus === 'API OFFLINE') {
          requestPPO(s);
        }

        if (s.ppoActionReady) {
          s.ppoActionReady = false;
          s.damageStats.ppoDecisions += 1;
          const action = s.ppoAction;
          const target = action === 1 || action === 3 ? s.g : s.p;
          const targetName = action === 1 || action === 3 ? 'GUARDIAN' : 'PLAYER';

          if (action === 0 || action === 1) {
            if (action === 1 && s.guardian <= 0) {
              s.enemyDecision = 'PPO: GUARDIAN DISABLED';
            } else {
              moveToward(s.e, target, 0.65, 1);
              s.enemyDecision = `PPO: APPROACH ${targetName}`;
            }
          } else if (action === 2 || action === 3) {
            s.enemyDecision = `PPO: ATTACK ${targetName}`;

            if (s.aiSteps >= s.nextAttackStep &&
                distance(s.e, target) <= 1.8 &&
                (action === 2 || s.guardian > 0)) {
              s.nextAttackStep = s.aiSteps + 2;

              if (action === 2) {
                const dealt = Math.min(20, s.player);
          s.player = Math.max(0, s.player - 20);
          s.damageStats.adversary += dealt;
                s.enemyHits += 1;
                s.message = 'PPO Adversary hit Player for 20 damage';
              } else {
                const dealt = Math.min(20, s.guardian);
          s.guardian = Math.max(0, s.guardian - 20);
          s.damageStats.adversary += dealt;
                s.message = 'PPO Adversary hit Guardian for 20 damage';
              }
            }
          } else if (action === 4) {
            const escapeFrom = s.powerStrikeActive && s.guardian > 0 ? s.g : s.p;
            const dx = s.e.x - escapeFrom.x;
            const dz = s.e.z - escapeFrom.z;
            const length = Math.hypot(dx, dz);

            if (length > 0.001) {
              s.e.x += dx / length * 0.75;
              s.e.z += dz / length * 0.75;
            }

            s.enemyDecision = 'PPO: RETREAT';
          }

          clampArena(s.e);
        }
      }

      if (s.guardian > 0 && s.player > 0 && s.enemy > 0 && !s.powerStrikeActive) {
        const playerThreat = distance(s.e, s.p);
        const guardianEnemyDistance = distance(s.g, s.e);

        // Protector: intercept nearby threats; otherwise stay close to Player.
        if (playerThreat < 5.0) {
          if (guardianEnemyDistance > 1.45) {
            moveToward(s.g, s.e, 2.7, dt);
          }
          s.guardianDecision = 'PROTECT: INTERCEPT';
        } else {
          if (distance(s.g, s.p) > 2.5) {
            moveToward(s.g, s.p, 2.7, dt);
          }
          s.guardianDecision = 'PROTECT: GUARD PLAYER';
        }

        if (distance(s.g, s.e) <= 1.45 && s.elapsed >= s.guardianAttack) {
          s.guardianAttack = s.elapsed + 1.4;
          const dealt = Math.min(6, s.enemy);
          s.enemy = Math.max(0, s.enemy - 6);
          s.damageStats.guardian += dealt;
          s.message = 'Guardian defended Player: 6 damage';
        }

        clampArena(s.g);
      } else if (s.guardian <= 0) s.guardianDecision = 'DISABLED';

      if (s.enemy <= 0) { s.enemy = 0; s.terminal = true; s.enemyDecision = 'DEFEATED'; s.message = 'VICTORY — Adversary defeated!'; }
      else if (s.player <= 0) { s.player = 0; s.terminal = true; s.message = 'MISSION FAILED — Player defeated'; }
    }
    playerMesh.current?.position.copy(s.p);
    guardianMesh.current?.position.copy(s.g);
    enemyMesh.current?.position.copy(s.e);
    if (s.elapsed - lastPublish.current > .15 || s.terminal && lastPublish.current !== -1) {
      lastPublish.current = s.terminal ? -1 : s.elapsed;
      onStats({ player: s.player, guardian: s.guardian, enemy: s.enemy, message: s.message, guardianDecision: s.guardianDecision, enemyDecision: s.enemyDecision, playerHits: s.playerHits, enemyHits: s.enemyHits, ppoStatus: s.ppoStatus, damageStats: { ...s.damageStats } });
    }
  });
  return <>
    <ambientLight intensity={.65}/><directionalLight position={[5, 12, 5]} intensity={2}/>
    <pointLight position={[-4, 3, -4]} intensity={30} color="#00bfff"/>
    <pointLight position={[4, 3, -4]} intensity={30} color="#ff3355"/>
    <Stars radius={70} depth={35} count={1500} factor={3} fade speed={.4}/>
    <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow><planeGeometry args={[20, 20]}/><meshStandardMaterial color="#10182a" metalness={.65} roughness={.45}/></mesh>
    <gridHelper args={[20, 20, '#23577a', '#1c2a44']} position={[0, .015, 0]}/>
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, .02, 0]}><ringGeometry args={[8.7, 8.82, 80]}/><meshBasicMaterial color="#2c82a9" side={THREE.DoubleSide}/></mesh>
    <group ref={playerMesh}><Character color="#d9e5ff" label="PLAYER"/></group>
    <group ref={guardianMesh}><Character color="#00bfff" label="GUARDIAN"/></group>
    <group ref={enemyMesh}><Character color="#ff3559" label="ADVERSARY"/></group>
    <OrbitControls makeDefault enablePan={false} minDistance={8} maxDistance={25} maxPolarAngle={Math.PI / 2.2} target={[0, 0, 0]}/>
  </>;
}
function Meter({ name, value, max, color }) {
  const percentage = Math.max(0, Math.min(100, (value / max) * 100));
  return <div className="meter"><div className="meter-label"><span>{name}</span><b>{value}/{max}</b></div><div className="meter-track"><div style={{ width: `${percentage}%`, background: color }}/></div></div>;
}
function App() {
  const keys = useRef(new Set());
  const [stats, setStats] = useState(START), [resetId, setResetId] = useState(0), [help, setHelp] = useState(true);
  useEffect(() => {
    const down = e => { if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) e.preventDefault(); keys.current.add(e.key.toLowerCase()); };
    const up = e => keys.current.delete(e.key.toLowerCase());
    const blur = () => keys.current.clear();
    window.addEventListener('keydown', down); window.addEventListener('keyup', up); window.addEventListener('blur', blur);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); window.removeEventListener('blur', blur); };
  }, []);
  return <main><header><div><span className="eyebrow">AI GAME EXPERIMENT · DAY 3</span><h1>PROJECT <strong>SENTINEL</strong></h1></div><span className="status">● SYSTEM ONLINE</span></header>
    <Canvas camera={{ position: [11, 12, 14], fov: 48 }} shadows><Arena keys={keys} resetId={resetId} onStats={setStats}/></Canvas>
    <aside className="hud"><Meter name="PLAYER" value={stats.player} max={100} color="#d9e5ff"/><Meter name="GUARDIAN" value={stats.guardian} max={120} color="#00bfff"/><Meter name="ADVERSARY" value={stats.enemy} max={140} color="#ff3559"/></aside>
    <section className="panel"><h2>MISSION 003</h2><p>{stats.message}</p><p className="muted">WASD / arrows: move · SPACE: attack (close range)</p><button onClick={() => { keys.current.clear(); setResetId(v => v + 1); }}>RESTART MISSION</button> <button onClick={() => setHelp(v => !v)}>{help ? 'HIDE' : 'SHOW'} INFO</button>{help && <p className="hint">Adversary decisions come from a trained PPO v4 model via a local Python API. The model is not learning during gameplay.</p>}
      <div className="ai-monitor"><h3>AI DECISION MONITOR</h3><p><b>Guardian:</b> {stats.guardianDecision}</p><p><b>Adversary:</b> {stats.enemyDecision} ? {stats.ppoStatus || 'CONNECTING'}</p><p className="muted">
      Damage ? Player: {stats.damageStats?.player ?? 0}
      {' ? '}Guardian: {stats.damageStats?.guardian ?? 0}
      {' ? '}Adversary: {stats.damageStats?.adversary ?? 0}
      {' ? '}Power Strikes: {stats.damageStats?.powerStrikes ?? 0}
      {' ? '}PPO actions: {stats.damageStats?.ppoDecisions ?? 0}
    </p>
    <p className="muted">Player hits: {stats.playerHits} · Enemy hits on player: {stats.enemyHits}</p></div>
    </section>
    <footer>GUARDIAN <b className="blue">●</b> &nbsp; ADVERSARY <b className="red">●</b> <span>BUILD 0.3.0</span></footer></main>;
}
createRoot(document.getElementById('root')).render(<App/>);
