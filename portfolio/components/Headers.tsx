"use client";
import Link from "next/link";

export default function Headers() {
    return(
        <header className="w-full p-4 flex items-center justify-center bg-gray-800 text-white">
            <nav className="flex gap-6">
                <Link href="/">Home</Link>
                <Link href="about">about</Link>
                <Link href="projects">projects</Link>
                <Link href="contact">contact</Link>
            </nav>
        </header>
    )
}