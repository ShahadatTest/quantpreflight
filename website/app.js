const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const SEVERITIES = ['critical', 'high', 'medium', 'low'];
const INSTALL_HINT = 'You can still run it locally with the command above.';
const NEW_FILE = 'config.json';
const HUB = 'https://huggingface.co';
const HUB_FILES = ['config.json', 'generation_config.json', 'tokenizer_config.json', 'model.safetensors.index.json', 'hf_quant_config.json', 'quantize_config.json'];
const MAX_HEADER_BYTES = 64 * 1024 * 1024;
const SHARD_WORKERS = 8;

const data = JSON.parse(document.getElementById('site-data').textContent);
const source = document.getElementById('source');
const gutter = document.getElementById('gutter');
const report = document.getElementById('report');
const status = document.getElementById('status');
const fileLabel = document.getElementById('file');
const scanButton = document.getElementById('scan');
const tabs = document.getElementById('tabs');
const engineInput = document.getElementById('engine');
const gpuInput = document.getElementById('gpu');
const tpInput = document.getElementById('tp');
const weightsInput = document.getElementById('weights');
const repoInput = document.getElementById('repo');
const hubButton = document.getElementById('hub-scan');

const element = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const syncScroll = () => {
  gutter.scrollTop = source.scrollTop;
};

const renderGutter = () => {
  const count = source.value.split('\n').length;
  gutter.textContent = Array.from({ length: count }, (_, index) => index + 1).join('\n');
  syncScroll();
};

const readTarget = () => ({
  engine: engineInput.value,
  gpu: gpuInput.value,
  tp: Math.max(1, parseInt(tpInput.value, 10) || 1),
  weights_gib: Math.max(0, parseFloat(weightsInput.value) || 0),
});

const writeTarget = (target) => {
  engineInput.value = target.engine;
  gpuInput.value = target.gpu;
  tpInput.value = target.tp;
  weightsInput.value = target.weights_gib || '';
};

const sameTarget = (first, second) => JSON.stringify(first) === JSON.stringify(second);

const renderFinding = (finding) => {
  const article = element('article', 'finding');
  const header = element('header');
  const rule = element('a', '', finding.rule_id);
  rule.href = `#${finding.rule_id}`;
  header.append(element('span', `tag ${finding.severity}`, finding.severity.toUpperCase()), rule);

  const details = element('details');
  details.append(element('summary', '', 'How to fix it'), element('p', '', finding.remediation));
  article.append(header, element('h3', '', finding.title), element('p', '', finding.message), details);
  return article;
};

const renderVerdict = (result) => {
  const count = result.findings.length;
  const totals = SEVERITIES
    .map((severity) => [result.findings.filter((finding) => finding.severity === severity).length, severity])
    .filter(([total]) => total);
  const failing = result.findings.some((finding) => ['critical', 'high'].includes(finding.severity));
  const stamp = failing ? ['fails', 'No-go'] : count ? ['passes', 'Caution'] : ['clean', 'Go'];
  let summary = result.tensors
    ? "A clean report doesn't prove the model will load or answer well."
    : 'This check read config.json only. Enter a model ID to check its tensors too.';
  if (count) summary = totals.map(([total, severity]) => `${total} ${severity}`).join(', ');
  if (count && !failing) summary += '. Below the default CI failure level.';

  const verdict = element('div', 'verdict');
  const title = count ? `${count} issue${count === 1 ? '' : 's'} found` : 'No issues found';
  verdict.append(
    element('p', 'label kind', result.quant),
    element('h2', '', title),
    element('span', `stamp ${stamp[0]}`, stamp[1]),
    element('p', '', summary),
  );
  if (result.vram_gib) verdict.append(element('p', 'vram', `Estimated ${result.vram_gib.toFixed(2)} GiB per GPU, for one request of up to 4,096 tokens.`));
  if (result.tensors) verdict.append(element('p', 'vram', `${result.tensors} tensors, ${result.weights_gib} GiB of weights. Headers read: ${result.bytes_read}.`));
  return verdict;
};

const SETUP_TITLES = {
  ready: 'Ready as asked',
  changed: 'It starts with these changes',
  blocked: 'No setting makes this start',
};

const copyButton = (text) => {
  const button = element('button', 'copy', 'Copy');
  button.type = 'button';
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = 'Copied';
    } catch {
      button.textContent = 'Select it';
      getSelection().selectAllChildren(button.parentElement.querySelector('code'));
    }
    setTimeout(() => { button.textContent = 'Copy'; }, 1600);
  });
  return button;
};

const renderSetup = (setup) => {
  const section = element('section', `setup ${setup.status}`);
  section.append(element('p', 'label kind', 'Setup'), element('h3', '', SETUP_TITLES[setup.status]));
  if (setup.changes.length) {
    const list = element('ul');
    setup.changes.forEach((change) => list.append(element('li', '', change)));
    section.append(list);
  }
  if (setup.status === 'blocked') {
    section.append(element('p', '', `${setup.blocking.join(', ')} needs a different checkpoint, GPU, or engine. quantpreflight suggest finds one that works.`));
  } else if (setup.max_context && setup.context_len) {
    const fits = setup.max_context > setup.context_len ? `, and up to ${setup.max_context} would fit` : '';
    section.append(element('p', '', `Runs with ${setup.context_len} tokens per request at tp=${setup.tp}${fits}.`));
  }
  if (setup.command) {
    const command = element('div', 'command');
    command.append(element('code', '', setup.command), copyButton(setup.command));
    section.append(command);
  }
  return section;
};

const renderResult = (result) => {
  report.classList.remove('stale');
  if (result.error) {
    const verdict = element('div', 'verdict');
    verdict.append(element('p', 'label kind', 'Not checked'), element('h2', '', "Couldn't read this"), element('p', '', result.error));
    report.replaceChildren(verdict);
    return;
  }
  const setup = result.setup ? [renderSetup(result.setup)] : [];
  report.replaceChildren(renderVerdict(result), ...setup, ...result.findings.map(renderFinding));
  report.scrollTop = 0;
};

const renderEmpty = (title, text) => {
  report.classList.remove('stale');
  const verdict = element('div', 'verdict');
  verdict.append(element('p', 'label kind', 'Ready'), element('h2', '', title), element('p', '', text));
  report.replaceChildren(verdict);
};

const formatBytes = (bytes) => (bytes < 1048576 ? `${(bytes / 1024).toFixed(0)} KiB` : `${(bytes / 1048576).toFixed(1)} MiB`);

const hubError = (response, name) => {
  if (response.status === 401 || response.status === 403) return new Error("This model is private or gated. Use the command line with HF_TOKEN.");
  return new Error(`Hugging Face answered ${response.status} for ${name}.`);
};

const fetchHubFile = async (url, name, counter) => {
  const response = await fetch(url);
  if (response.status === 404) return null;
  if (!response.ok) throw hubError(response, name);
  const text = await response.text();
  counter.bytes += text.length;
  return text;
};

const fetchShardHeader = async (url, name, counter) => {
  const first = await fetch(url, { headers: { Range: 'bytes=0-7' } });
  if (first.status === 404) return null;
  if (!first.ok) throw hubError(first, name);
  const prefix = await first.arrayBuffer();
  const length = new DataView(prefix).getBigUint64(0, true);
  if (length === 0n || length > BigInt(MAX_HEADER_BYTES)) throw new Error(`${name} has an invalid header size.`);
  const body = await fetch(url, { headers: { Range: `bytes=8-${7n + length}` } });
  if (!body.ok) throw hubError(body, name);
  const text = await body.text();
  counter.bytes += 8 + text.length;
  return text;
};

const shardNames = (index, siblings) => {
  if (index) return [...new Set(Object.values(JSON.parse(index).weight_map || {}))].sort();
  if (siblings.includes('model.safetensors')) return ['model.safetensors'];
  return siblings.filter((name) => name.toLowerCase().endsWith('.safetensors'));
};

const fetchRepo = async (repo) => {
  const api = await fetch(`${HUB}/api/models/${repo}`);
  if (api.status === 401 || api.status === 404) throw new Error("That model wasn't found. It may be private or gated.");
  if (!api.ok) throw hubError(api, repo);
  const info = await api.json();
  const siblings = (info.siblings || []).map((item) => item.rfilename);
  if (info.gated) throw new Error('This model is gated. Use the command line with HF_TOKEN.');

  const base = `${HUB}/${repo}/resolve/main`;
  const counter = { bytes: 0 };
  const texts = await Promise.all(HUB_FILES.map((name) => fetchHubFile(`${base}/${name}`, name, counter)));
  const files = Object.fromEntries(HUB_FILES.map((name, position) => [name, texts[position]]));
  const names = shardNames(files['model.safetensors.index.json'], siblings);
  if (!names.length) {
    const hasGguf = siblings.some((name) => name.toLowerCase().endsWith('.gguf'));
    throw new Error(hasGguf ? 'This is a GGUF model. Use the command line to scan it.' : 'This model has no Safetensors files.');
  }

  const shards = {};
  const queue = [...names];
  await Promise.all(Array.from({ length: Math.min(SHARD_WORKERS, names.length) }, async () => {
    while (queue.length) {
      const name = queue.shift();
      shards[name] = await fetchShardHeader(`${base}/${name}`, name, counter);
    }
  }));
  const ordered = Object.fromEntries(names.map((name) => [name, shards[name]]));
  return { payload: { repo, files, shards: ordered, siblings }, bytes: counter.bytes };
};

let engine;
const loadEngine = () => {
  engine ??= (async () => {
    await new Promise((resolve, reject) => {
      const script = element('script');
      script.src = `${PYODIDE}pyodide.js`;
      script.onload = resolve;
      script.onerror = reject;
      document.head.append(script);
    });
    const python = await loadPyodide({ indexURL: PYODIDE });
    const scanner = await fetch('scanner.zip');
    if (!scanner.ok) throw new Error('scanner.zip is missing');
    python.unpackArchive(await scanner.arrayBuffer(), 'zip', { extractDir: '/home/pyodide/scanner' });
    python.runPython(`
import sys
sys.path.insert(0, '/home/pyodide/scanner')
`);
    return python.pyimport('site_scan');
  })().catch((error) => {
    engine = undefined;
    throw error;
  });
  return engine;
};

const scanHub = async (event) => {
  event.preventDefault();
  const repo = repoInput.value.trim().replace(/^https:\/\/(huggingface\.co|hf\.co)\//, '').replace(/\/$/, '');
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) {
    renderResult({ error: 'Enter a model as org/name, such as Qwen/Qwen2.5-7B-Instruct-AWQ.' });
    return;
  }
  const target = readTarget();
  hubButton.disabled = true;
  scanButton.disabled = true;
  status.textContent = `Reading headers of ${repo}…`;
  try {
    const { payload, bytes } = await fetchRepo(repo);
    status.textContent = engine ? 'Checking…' : 'Loading the scanner. This is about 6 MB the first time.';
    const python = await loadEngine();
    const result = JSON.parse(python.scan_repo(JSON.stringify(payload), target.engine, target.gpu, target.tp));
    if (result.config) {
      source.value = result.config;
      fileLabel.textContent = `${repo} config.json`;
      weightsInput.value = result.weights_gib;
      renderGutter();
      selectTab('');
    }
    result.bytes_read = formatBytes(bytes);
    renderResult(result);
    status.textContent = '';
  } catch (error) {
    console.error(error);
    status.textContent = '';
    renderResult({ error: error.message });
  } finally {
    hubButton.disabled = false;
    scanButton.disabled = false;
  }
};

const scan = async () => {
  const text = source.value;
  const target = readTarget();
  status.textContent = '';
  if (!text.trim()) {
    renderEmpty('Nothing to check', 'Paste a config.json on this side.');
    return;
  }
  const known = data.examples.find((example) => example.text === text && sameTarget(example.target, target));
  if (known) {
    renderResult(known.result);
    return;
  }

  scanButton.disabled = true;
  status.textContent = engine ? 'Checking…' : 'Loading the scanner. This is about 6 MB the first time.';
  try {
    const python = await loadEngine();
    const result = JSON.parse(python.scan_text(text, target.engine, target.gpu, target.tp, target.weights_gib));
    const current = source.value === text && sameTarget(readTarget(), target);
    if (current) renderResult(result);
    status.textContent = current ? '' : 'The model changed. Check it again.';
  } catch (error) {
    console.error(error);
    status.textContent = `Couldn't load the scanner. ${INSTALL_HINT}`;
  } finally {
    scanButton.disabled = false;
  }
};

const selectTab = (slug) => {
  tabs.querySelectorAll('button').forEach((button) => button.setAttribute('aria-pressed', button.dataset.slug === slug));
};

const showExample = (example) => {
  source.value = example.text;
  source.scrollTop = 0;
  source.scrollLeft = 0;
  fileLabel.textContent = example.file;
  status.textContent = '';
  writeTarget(example.target);
  selectTab(example.slug);
  renderGutter();
  renderResult(example.result);
};

const markChanged = () => {
  report.classList.add('stale');
  status.textContent = 'Press Check to see the new result.';
  selectTab('');
};

data.engines.forEach(({ id, name }) => engineInput.append(new Option(name, id)));
gpuInput.append(new Option('No GPU named', ''));
data.gpus.forEach(({ id, label }) => gpuInput.append(new Option(label, id)));

data.examples.forEach((example) => {
  const button = element('button', '', example.label);
  button.type = 'button';
  button.dataset.slug = example.slug;
  button.addEventListener('click', () => showExample(example));
  tabs.append(button);
});

source.addEventListener('scroll', syncScroll);
source.addEventListener('input', () => {
  renderGutter();
  fileLabel.textContent = NEW_FILE;
  markChanged();
});
source.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) scan();
});
[engineInput, gpuInput, tpInput, weightsInput].forEach((input) => input.addEventListener('input', markChanged));
scanButton.addEventListener('click', scan);
document.getElementById('hub').addEventListener('submit', scanHub);
document.getElementById('clear').addEventListener('click', () => {
  source.value = '';
  fileLabel.textContent = NEW_FILE;
  status.textContent = '';
  selectTab('');
  renderGutter();
  renderEmpty('Paste a config.json', 'It runs here in your browser. Nothing is uploaded.');
  source.focus();
});

document.querySelectorAll('.copy').forEach((button) => button.addEventListener('click', async () => {
  const from = button.dataset.copyFrom;
  const text = from ? document.getElementById(from).textContent : button.dataset.copy;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = 'Copied';
  } catch {
    button.textContent = 'Select it';
    getSelection().selectAllChildren(button.parentElement.querySelector('code'));
  }
  setTimeout(() => { button.textContent = 'Copy'; }, 1600);
}));

window.addEventListener('resize', syncScroll);
showExample(data.examples[0]);
