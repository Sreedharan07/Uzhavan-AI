using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Diagnostics;
using System.Threading;
using System.Text;

namespace UzhavanAI
{
    class Program
    {
        static Process backendProcess = null;
        static int activePort = 8000;
        static string projectRoot = "";
        static HttpListener fallbackServer = null;
        static Thread fallbackThread = null;
        static volatile bool isRunning = true;

        static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.Title = "🌾 UZHAVAN AI - Desktop Unified Runner";

            // Determine project root directory
            projectRoot = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\', '/');
            if (!File.Exists(Path.Combine(projectRoot, "app.py")))
            {
                string cur = Directory.GetCurrentDirectory();
                if (File.Exists(Path.Combine(cur, "app.py")))
                {
                    projectRoot = cur;
                }
            }

            PrintBanner();

            // Set up exit hooks
            Console.CancelKeyPress += (s, e) => {
                e.Cancel = true;
                Shutdown();
            };
            AppDomain.CurrentDomain.ProcessExit += (s, e) => Shutdown();

            // Find an open port
            activePort = FindAvailablePort("127.0.0.1", 8000);

            // Locate Python / VirtualEnv
            string pythonExe = FindPythonExecutable();

            if (!string.IsNullOrEmpty(pythonExe) && File.Exists(Path.Combine(projectRoot, "app.py")))
            {
                StartPythonBackend(pythonExe, activePort);
            }
            else
            {
                StartFallbackServer(activePort);
            }

            // Open browser after brief delay
            Thread.Sleep(1500);
            string appUrl = "http://localhost:" + activePort;
            OpenUrl(appUrl);

            PrintMenu();

            // Check if console input is available
            bool canReadKey = true;
            try
            {
                bool test = Console.KeyAvailable;
            }
            catch
            {
                canReadKey = false;
            }

            // Interactive keyboard loop
            while (isRunning)
            {
                if (canReadKey)
                {
                    try
                    {
                        if (Console.KeyAvailable)
                        {
                            ConsoleKeyInfo key = Console.ReadKey(true);
                            switch (key.Key)
                            {
                                case ConsoleKey.D1:
                                case ConsoleKey.NumPad1:
                                    Console.WriteLine("\n🌐 Opening Web Application: " + appUrl);
                                    OpenUrl(appUrl);
                                    break;
                                case ConsoleKey.D2:
                                case ConsoleKey.NumPad2:
                                    Console.WriteLine("\n📚 Opening Swagger API Docs: " + appUrl + "/docs");
                                    OpenUrl(appUrl + "/docs");
                                    break;
                                case ConsoleKey.D3:
                                case ConsoleKey.NumPad3:
                                    Console.WriteLine("\n🔬 Opening Research Benchmark: " + appUrl + "/api/research/benchmark-summary");
                                    OpenUrl(appUrl + "/api/research/benchmark-summary");
                                    break;
                                case ConsoleKey.D4:
                                case ConsoleKey.NumPad4:
                                    OpenPaperPdf();
                                    break;
                                case ConsoleKey.D5:
                                case ConsoleKey.NumPad5:
                                    OpenPaperPack();
                                    break;
                                case ConsoleKey.Q:
                                    Console.WriteLine("\n👋 Shutting down UZHAVAN AI...");
                                    isRunning = false;
                                    Shutdown();
                                    return;
                                default:
                                    break;
                            }
                        }
                    }
                    catch
                    {
                        canReadKey = false;
                    }
                }
                Thread.Sleep(200);
            }
        }

        static void PrintBanner()
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(@"
======================================================================
🌾 UZHAVAN AI — Rural Tamil Dialect Voice Assistant
   Standalone Windows Project Launcher (Research & Paper Edition)
======================================================================
   Project Root: " + projectRoot + @"
======================================================================");
            Console.ResetColor();
        }

        static void PrintMenu()
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(@"
----------------------------------------------------------------------
 Interactive Control Menu (Press any key below):
   [1] Open Web Application (Browser: http://localhost:" + activePort + @")
   [2] Open Swagger API Interactive Docs (http://localhost:" + activePort + @"/docs)
   [3] Open Research Benchmark Summary JSON
   [4] Open IEEE Research Paper PDF
   [5] Open ChatGPT Paper Modeling Pack (Markdown)
   [Q] Quit and Stop Servers
----------------------------------------------------------------------
 Live Application Logs:
");
            Console.ResetColor();
        }

        static string FindPythonExecutable()
        {
            // 1. Check project .venv
            string venvPy = Path.Combine(projectRoot, @".venv\Scripts\python.exe");
            if (File.Exists(venvPy)) return venvPy;

            // 2. Check system PATH python
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo("where", "python.exe")
                {
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                using (Process p = Process.Start(psi))
                {
                    string outStr = p.StandardOutput.ReadToEnd();
                    p.WaitForExit();
                    string[] lines = outStr.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                    if (lines.Length > 0 && File.Exists(lines[0])) return lines[0];
                }
            }
            catch { }

            return null;
        }

        static void StartPythonBackend(string pythonExe, int port)
        {
            try
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("🚀 Launching Backend & Web UI on http://localhost:" + port + "...");
                Console.WriteLine("   Python runtime: " + pythonExe);
                Console.ResetColor();

                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = "app.py --port " + port + " --no-browser",
                    WorkingDirectory = projectRoot,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    StandardOutputEncoding = Encoding.UTF8,
                    StandardErrorEncoding = Encoding.UTF8
                };

                backendProcess = new Process { StartInfo = psi };
                backendProcess.OutputDataReceived += (s, e) => {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        Console.WriteLine("   " + e.Data);
                    }
                };
                backendProcess.ErrorDataReceived += (s, e) => {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        Console.WriteLine("   [err] " + e.Data);
                    }
                };

                backendProcess.Start();
                backendProcess.BeginOutputReadLine();
                backendProcess.BeginErrorReadLine();
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("⚠️ Could not launch Python backend directly: " + ex.Message);
                Console.WriteLine("   Falling back to built-in HTTP server...");
                Console.ResetColor();
                StartFallbackServer(port);
            }
        }

        static void StartFallbackServer(int port)
        {
            try
            {
                fallbackServer = new HttpListener();
                fallbackServer.Prefixes.Add("http://localhost:" + port + "/");
                fallbackServer.Prefixes.Add("http://127.0.0.1:" + port + "/");
                fallbackServer.Start();

                fallbackThread = new Thread(() => {
                    while (isRunning && fallbackServer != null && fallbackServer.IsListening)
                    {
                        try
                        {
                            var ctx = fallbackServer.GetContext();
                            ThreadPool.QueueUserWorkItem((state) => HandleFallbackRequest(ctx));
                        }
                        catch { }
                    }
                });
                fallbackThread.IsBackground = true;
                fallbackThread.Start();

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("✅ Built-in lightweight server active on http://localhost:" + port);
                Console.ResetColor();
            }
            catch (Exception ex)
            {
                Console.WriteLine("⚠️ Built-in server error: " + ex.Message);
            }
        }

        static void HandleFallbackRequest(HttpListenerContext ctx)
        {
            try
            {
                string rawUrl = ctx.Request.Url.AbsolutePath;
                string distDir = Path.Combine(projectRoot, @"frontend\dist");

                // API mocks
                if (rawUrl == "/api/health")
                {
                    SendJson(ctx, "{\"message\":\"🌾 UZHAVAN AI - Standalone Runner\",\"status\":\"running\"}");
                    return;
                }
                if (rawUrl == "/api/model-status")
                {
                    SendJson(ctx, "{\"status\":\"running\",\"device\":\"cpu\",\"asr_loaded\":true,\"dialect_loaded\":true,\"intent_loaded\":true,\"crop_loaded\":true,\"gpu_name\":\"N/A\",\"vram\":\"N/A\"}");
                    return;
                }
                if (rawUrl == "/api/research/benchmark-summary")
                {
                    string jsonPath = Path.Combine(projectRoot, @"results\research\acoustic_robustness_results.json");
                    if (File.Exists(jsonPath))
                    {
                        SendJson(ctx, File.ReadAllText(jsonPath, Encoding.UTF8));
                        return;
                    }
                }

                // Serve static dist files
                string rel = rawUrl.TrimStart('/');
                if (string.IsNullOrEmpty(rel)) rel = "index.html";
                string targetFile = Path.Combine(distDir, rel);

                if (!File.Exists(targetFile))
                {
                    targetFile = Path.Combine(distDir, "index.html");
                }

                if (File.Exists(targetFile))
                {
                    byte[] data = File.ReadAllBytes(targetFile);
                    string ext = Path.GetExtension(targetFile).ToLower();
                    if (ext == ".html") ctx.Response.ContentType = "text/html; charset=utf-8";
                    else if (ext == ".js") ctx.Response.ContentType = "application/javascript";
                    else if (ext == ".css") ctx.Response.ContentType = "text/css";
                    else if (ext == ".svg") ctx.Response.ContentType = "image/svg+xml";
                    else if (ext == ".png") ctx.Response.ContentType = "image/png";
                    else if (ext == ".json") ctx.Response.ContentType = "application/json";

                    ctx.Response.ContentLength64 = data.Length;
                    ctx.Response.OutputStream.Write(data, 0, data.Length);
                }
                else
                {
                    ctx.Response.StatusCode = 404;
                }
            }
            catch { }
            finally
            {
                try { ctx.Response.OutputStream.Close(); } catch { }
            }
        }

        static void SendJson(HttpListenerContext ctx, string json)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(json);
            ctx.Response.ContentType = "application/json; charset=utf-8";
            ctx.Response.ContentLength64 = bytes.Length;
            ctx.Response.OutputStream.Write(bytes, 0, bytes.Length);
        }

        static int FindAvailablePort(string host, int startPort)
        {
            for (int p = startPort; p < startPort + 20; p++)
            {
                TcpListener l = null;
                try
                {
                    l = new TcpListener(IPAddress.Parse(host), p);
                    l.Start();
                    l.Stop();
                    return p;
                }
                catch { }
                finally
                {
                    if (l != null)
                    {
                        try { l.Stop(); } catch { }
                    }
                }
            }
            return startPort;
        }

        static void OpenUrl(string url)
        {
            try
            {
                Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            }
            catch (Exception ex)
            {
                Console.WriteLine("⚠️ Could not launch browser automatically: " + ex.Message);
            }
        }

        static void OpenPaperPdf()
        {
            string pdfPath = Path.Combine(projectRoot, @"research\UZHAVAN_AI_IEEE_Research_Paper.pdf");
            if (File.Exists(pdfPath))
            {
                Console.WriteLine("\n📄 Opening IEEE Research Paper PDF: " + pdfPath);
                OpenUrl(pdfPath);
            }
            else
            {
                Console.WriteLine("\n⚠️ PDF not found at " + pdfPath);
            }
        }

        static void OpenPaperPack()
        {
            string packPath = Path.Combine(projectRoot, "UZHAVAN_AI_Paper_Modeling_Pack.md");
            if (File.Exists(packPath))
            {
                Console.WriteLine("\n📝 Opening ChatGPT Paper Modeling Pack: " + packPath);
                OpenUrl(packPath);
            }
            else
            {
                Console.WriteLine("\n⚠️ Pack not found at " + packPath);
            }
        }

        static void Shutdown()
        {
            try
            {
                if (backendProcess != null && !backendProcess.HasExited)
                {
                    // Kill process tree
                    ProcessStartInfo taskkill = new ProcessStartInfo("taskkill", "/F /T /PID " + backendProcess.Id)
                    {
                        CreateNoWindow = true,
                        UseShellExecute = false
                    };
                    Process p = Process.Start(taskkill);
                    if (p != null) p.WaitForExit(2000);
                }
            }
            catch { }

            try
            {
                if (fallbackServer != null && fallbackServer.IsListening)
                {
                    fallbackServer.Stop();
                    fallbackServer.Close();
                }
            }
            catch { }
        }
    }
}
