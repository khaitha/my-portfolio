"use client";
import React, { useEffect } from "react";

interface Props {
    speed: number;
    backgroundColor: string;
    starColor: [number, number, number];
    starCount: number;
}

type ShootingStar = {
    x: number;
    y: number;
    vx: number;
    vy: number;
    length: number;
    life: number;
    maxLife: number;
};

function randomDirection() {
    // Returns a random direction: right, diagonal, etc.
    const angle = Math.random() * Math.PI * 2;
    const speed = 8 + Math.random() * 4;
    return {
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
    };
}

function createShootingStar(w: number, h: number): ShootingStar {
    // Start from a random edge
    let x = Math.random() * w;
    let y = Math.random() * h;
    // Prefer starting from left/top edges
    if (Math.random() < 0.5) x = 0;
    if (Math.random() < 0.5) y = 0;
    const { vx, vy } = randomDirection();
    return {
        x,
        y,
        vx,
        vy,
        length: 80 + Math.random() * 40,
        life: 0,
        maxLife: 80 + Math.random() * 40,
    };
}

export default function Starfield(props: Props) {
    const {
        speed = 0.05,
        backgroundColor = "black",
        starColor = [255, 255, 255],
        starCount = 3000,
    } = props;

    useEffect(() => {
        const canvas = document.getElementById("starfield") as HTMLCanvasElement;

        if (canvas) {
            const c = canvas.getContext("2d");

            if (c) {
                let w = window.innerWidth;
                let h = window.innerHeight;

                const setCanvasExtents = () => {
                    canvas.width = w;
                    canvas.height = h;
                };

                setCanvasExtents();

                window.onresize = () => {
                    w = window.innerWidth;
                    h = window.innerHeight;
                    setCanvasExtents();
                };

                const makeStars = (count: number) => {
                    const out = [];
                    for (let i = 0; i < count; i++) {
                        const s = {
                            x: Math.random() * 1600 - 800,
                            y: Math.random() * 900 - 450,
                            z: Math.random() * 1000,
                        };
                        out.push(s);
                    }
                    return out;
                };

                let stars = makeStars(starCount);

                // Shooting stars
                const shootingStarCount = 3;
                let shootingStars: ShootingStar[] = [];
                for (let i = 0; i < shootingStarCount; i++) {
                    shootingStars.push(createShootingStar(w, h));
                }

                const clear = () => {
                    c.fillStyle = backgroundColor;
                    c.fillRect(0, 0, canvas.width, canvas.height);
                };

                const putPixel = (x: number, y: number, brightness: number) => {
                    const rgb =
                        "rgba(" +
                        starColor[0] +
                        "," +
                        starColor[1] +
                        "," +
                        starColor[2] +
                        "," +
                        brightness +
                        ")";
                    c.fillStyle = rgb;
                    c.fillRect(x, y, 1, 1);
                };

                const moveStars = (distance: number) => {
                    const count = stars.length;
                    for (var i = 0; i < count; i++) {
                        const s = stars[i];
                        s.z -= distance;
                        while (s.z <= 1) {
                            s.z += 1000;
                        }
                    }
                };

                let prevTime: number;
                const init = (time: number) => {
                    prevTime = time;
                    requestAnimationFrame(tick);
                };

                const tick = (time: number) => {
                    let elapsed = time - prevTime;
                    prevTime = time;

                    moveStars(elapsed * speed);

                    clear();

                    const cx = w / 2;
                    const cy = h / 2;

                    const count = stars.length;
                    for (var i = 0; i < count; i++) {
                        const star = stars[i];

                        const x = cx + star.x / (star.z * 0.001);
                        const y = cy + star.y / (star.z * 0.001);

                        if (x < 0 || x >= w || y < 0 || y >= h) {
                            continue;
                        }

                        const d = star.z / 1000.0;
                        const b = 1 - d * d;

                        putPixel(x, y, b);
                    }

                    // Draw shooting stars
                    for (let i = 0; i < shootingStars.length; i++) {
                        let s = shootingStars[i];
                        // Draw as a line
                        c.save();
                        c.globalAlpha = 0.8 * (1 - s.life / s.maxLife);
                        c.strokeStyle = `rgba(${starColor[0]},${starColor[1]},${starColor[2]},1)`;
                        c.lineWidth = 2.5;
                        c.beginPath();
                        c.moveTo(s.x, s.y);
                        c.lineTo(s.x - s.vx * s.length / 20, s.y - s.vy * s.length / 20);
                        c.stroke();
                        c.restore();

                        // Move
                        s.x += s.vx;
                        s.y += s.vy;
                        s.life += 1;

                        // If out of bounds or life ended, respawn
                        if (
                            s.x < -100 ||
                            s.x > w + 100 ||
                            s.y < -100 ||
                            s.y > h + 100 ||
                            s.life > s.maxLife
                        ) {
                            shootingStars[i] = createShootingStar(w, h);
                        }
                    }

                    requestAnimationFrame(tick);
                };

                requestAnimationFrame(init);

                // add window resize listener:
                window.addEventListener("resize", function () {
                    w = window.innerWidth;
                    h = window.innerHeight;
                    setCanvasExtents();
                });
            } else {
                console.error("Could not get 2d context from canvas element");
            }
        } else {
            console.error('Could not find canvas element with id "starfield"');
        }

        return () => {
            window.onresize = null;
        };
    }, [starColor, backgroundColor, speed, starCount]);

    return (
        <canvas
            id="starfield"
            style={{
                padding: 0,
                margin: 0,
                position: "fixed",
                top: 0,
                right: 0,
                bottom: 0,
                left: 0,
                zIndex: 10,
                opacity: 1,
                pointerEvents: "none",
                mixBlendMode: "screen",
            }}
        ></canvas>
    );
}