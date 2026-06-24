'use strict';
const https = require('https');
const crypto = require('crypto');
const { URL } = require('url');

const CONFIG = {
  region: '${region}',
  userPoolId: '${user_pool_id}',
  clientId: '${client_id}',
  clientSecret: '${client_secret}',
  cognitoDomain: '${cognito_domain}',
  cookieName: 'TOKEN',
  scopes: 'openid email',
};

const JWKS_URL = `https://cognito-idp.$${CONFIG.region}.amazonaws.com/$${CONFIG.userPoolId}/.well-known/jwks.json`;
let jwksCache = null;

exports.handler = async (event) => {
  const request = event.Records[0].cf.request;
  const headers = request.headers;
  const host = headers.host[0].value;
  const uri = request.uri;
  const qs = request.querystring || '';
  const method = request.method;

  // CORS preflight — respond directly
  if (method === 'OPTIONS') {
    return {
      status: '204',
      headers: {
        'access-control-allow-origin': [{ value: `https://$${host}` }],
        'access-control-allow-methods': [{ value: 'GET,POST,PUT,DELETE,OPTIONS' }],
        'access-control-allow-headers': [{ value: 'Content-Type,Authorization,X-Requested-With,Accept' }],
        'access-control-allow-credentials': [{ value: 'true' }],
        'access-control-max-age': [{ value: '86400' }],
      },
    };
  }

  // Check for callback with auth code
  if (qs.includes('code=')) {
    const params = new URLSearchParams(qs);
    const code = params.get('code');
    const state = params.get('state') || '/';
    if (code) {
      try {
        const tokens = await exchangeCode(code, `https://$${host}`);
        return {
          status: '302',
          headers: {
            location: [{ value: `https://$${host}$${state}` }],
            'set-cookie': [
              { value: `$${CONFIG.cookieName}=$${tokens.id_token}; Path=/; Secure; HttpOnly; SameSite=Lax; Max-Age=604800` },
            ],
          },
        };
      } catch (e) {
        console.error('Token exchange failed:', e.message);
        return {
          status: '200',
          body: `Token exchange failed: $${e.message}`,
          headers: { 'content-type': [{ value: 'text/plain' }] },
        };
      }
    }
  }

  // Check existing token cookie
  const cookies = parseCookies(headers);
  const token = cookies[CONFIG.cookieName];
  if (token) {
    try {
      await verifyToken(token);
      return request; // authenticated, pass through
    } catch (e) {
      console.log('Token invalid:', e.message);
    }
  }

  // Redirect to Cognito login
  const redirectUri = `https://$${host}`;
  const state = uri === '/' ? '/' : uri;
  const authUrl = `https://$${CONFIG.cognitoDomain}/authorize?` +
    `response_type=code&client_id=$${CONFIG.clientId}` +
    `&redirect_uri=$${encodeURIComponent(redirectUri)}` +
    `&scope=$${CONFIG.scopes.replace(/ /g, '+')}` +
    `&state=$${encodeURIComponent(state)}`;

  return { status: '302', headers: { location: [{ value: authUrl }] } };
};

async function exchangeCode(code, redirectUri) {
  const body = new URLSearchParams({
    grant_type: 'authorization_code',
    code,
    redirect_uri: redirectUri,
    client_id: CONFIG.clientId,
  }).toString();

  const auth = Buffer.from(`$${CONFIG.clientId}:$${CONFIG.clientSecret}`).toString('base64');
  const data = await httpsPost(`https://$${CONFIG.cognitoDomain}/oauth2/token`, body, {
    'Content-Type': 'application/x-www-form-urlencoded',
    Authorization: `Basic $${auth}`,
  });
  return JSON.parse(data);
}

async function verifyToken(token) {
  const [headerB64, payloadB64, signatureB64] = token.split('.');
  if (!headerB64 || !payloadB64 || !signatureB64) throw new Error('Malformed JWT');

  const payload = JSON.parse(Buffer.from(payloadB64, 'base64url').toString());
  const now = Math.floor(Date.now() / 1000);
  if (payload.exp && payload.exp < now) throw new Error('Token expired');
  if (payload.iss !== `https://cognito-idp.$${CONFIG.region}.amazonaws.com/$${CONFIG.userPoolId}`) {
    throw new Error('Invalid issuer');
  }
  return payload;
}

function parseCookies(headers) {
  const cookies = {};
  if (headers.cookie) {
    for (const cookieHeader of headers.cookie) {
      for (const part of cookieHeader.value.split(';')) {
        const [k, ...v] = part.trim().split('=');
        cookies[k] = v.join('=');
      }
    }
  }
  return cookies;
}

function httpsPost(url, body, headers) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const req = https.request({
      hostname: parsed.hostname,
      path: parsed.pathname + parsed.search,
      method: 'POST',
      headers: { ...headers, 'Content-Length': Buffer.byteLength(body) },
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        if (res.statusCode >= 400) reject(new Error(`HTTP $${res.statusCode}: $${data}`));
        else resolve(data);
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}
