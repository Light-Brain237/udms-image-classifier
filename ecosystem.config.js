module.exports = {
	apps: [
		{
			name: "udms-classifier",
			cwd: "/www/wwwroot/udms-image-classifier",
			script: "./.venv/bin/uvicorn",
			args: "app.main:app --host 127.0.0.1 --port 8000 --workers 2",
			interpreter: "none", // critical: don't run it through node
			env_file: "./.env", // PM2 v5+ reads this; otherwise inline env: {...}
			max_memory_restart: "1500M",
			out_file: "./logs/out.log",
			error_file: "./logs/err.log",
		},
	],
};
