"""Box2D proxy of GameScene.unity; scores are NOT Unity-verified."""

import math

import numpy as np
from Box2D import b2ContactListener, b2World


# Serialized values from GameScene.unity, PipeObject.prefab and Physics2DSettings.
RULES = {
    'dt': 0.02, 'gravity': -9.81 * 3.01, 'flap_velocity': 12.,
    'upward_cap': 10., 'cooldown': .15, 'pipe_speed': 4.79,
    'spawn_interval': 1.02, 'spawn_x': 4.34, 'gap_min': -2., 'gap_max': 2.,
    'pipe_scale_x': 2.0314608, 'pipe_scale_y': 2.2903721,
    'bird_scale': 1.8634515, 'bird_radius': .12671039 * 1.8634515,
}


class _Contacts(b2ContactListener):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def BeginContact(self, contact):
        fixtures = (contact.fixtureA, contact.fixtureB)
        if not any(f.body == self.env.bird for f in fixtures):
            return
        other = next(f for f in fixtures if f.body != self.env.bird)
        if other.sensor:
            if not other.body.userData['passed']:
                other.body.userData['passed'] = True
                self.env.pending_score += 1
        else:
            self.env.done = True


class HeadlessFlappyEnv:
    backend = 'box2d-proxy-v2'

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.world = b2World(gravity=(0, RULES['gravity']), doSleep=False)
        self.listener = _Contacts(self)
        self.world.contactListener = self.listener
        self.bird = self.world.CreateDynamicBody(position=(-2, 0), bullet=True)
        self.bird.CreateCircleFixture(
            radius=RULES['bird_radius'],
            pos=(.02 * RULES['bird_scale'], -.0014556028 * RULES['bird_scale']),
            density=1, friction=.2,
        )
        # Ground collider and scene boundary colliders, including local scaling.
        for x, y, hx, hy in [
            (-.00506413 * 2.4019132, -6.51 + .013125956 * 1.1820257,
             3.3598711 * 2.4019132 / 2, 1.1497363 * 1.1820257 / 2),
            (.01, 7.72, 5, .5), (-4.54, -.01, .5, 7.5), (4.54, -.01, .5, 7.5),
        ]:
            body = self.world.CreateStaticBody(position=(x, y))
            body.CreatePolygonFixture(box=(hx, hy))
        self.pipes = []
        self.cooldown = 0.
        self.spawn_ticks = 0
        self.score = 0
        self.done = False
        self.passed = False
        self.pending_score = 0
        self._spawn_pipe()
        return self._state(), {'score': 0, 'backend': self.backend}

    def _spawn_pipe(self):
        sx, sy = RULES['pipe_scale_x'], RULES['pipe_scale_y']
        body = self.world.CreateKinematicBody(
            position=(RULES['spawn_x'], float(self.rng.uniform(-2, 2))),
            linearVelocity=(-RULES['pipe_speed'], 0), userData={'passed': False},
        )
        for direction in (-1, 1):
            body.CreatePolygonFixture(box=(.52 * sx / 2, 3.2 * 1.5 * sy / 2,
                                          (0, direction * 3.1 * sy), 0))
        body.CreatePolygonFixture(box=(.38534224 * sx / 2, 1.4099658 * sy / 2,
                                      (.21 * sx, .002 * sy), 0), isSensor=True)
        self.pipes.append(body)
        # The original spawner checks zero before decrementing its 1s timer.
        self.spawn_ticks = 51

    def _state(self):
        nearby = [p for p in self.pipes if p.position.x + 2 > -1]
        pipe = min(nearby, key=lambda p: p.position.x) if nearby else None
        return np.array([
            self.bird.position.y / 5, self.bird.linearVelocity.y / 10,
            (pipe.position.x + 2) / 8 if pipe else 0,
            (pipe.position.y + .002 * RULES['pipe_scale_y']) / 5 if pipe else 0,
            self.cooldown / RULES['cooldown'],
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode finished; call reset before step')
        if action not in (0, 1):
            raise ValueError('action must be 0 or 1')
        dt = RULES['dt']
        self.passed = False
        self.pending_score = 0
        if action == 1 and self.cooldown <= 0:
            self.bird.linearVelocity = (0, RULES['flap_velocity'])
            self.cooldown = RULES['cooldown']
        self.world.Step(dt, 8, 3)
        # Reference cadence: FixedUpdate impulse -> physics -> Update rotation/cap.
        self.bird.angle = math.radians(self.bird.linearVelocity.y * 2)
        self.bird.angularVelocity = 0
        self.bird.linearVelocity = (0, min(self.bird.linearVelocity.y, RULES['upward_cap']))
        self.cooldown = max(0., self.cooldown - dt)
        if not self.done:
            self.score += self.pending_score
            self.passed = self.pending_score > 0
        self.spawn_ticks -= 1
        if self.spawn_ticks == 0 and not self.done:
            self._spawn_pipe()
        for pipe in self.pipes[:]:
            if pipe.position.x <= -5:
                self.world.DestroyBody(pipe)
                self.pipes.remove(pipe)
        state = self._state()
        if self.done:
            reward = -5.
        elif self.passed:
            reward = 10.
        else:
            reward = .1 - .1 * action
            if self.pipes:
                reward += .2 if abs(float(state[0] - state[3])) * 5 <= .8 else -.1
        return state, reward, self.done, {'score': self.score, 'backend': self.backend}

    def close(self):
        pass
